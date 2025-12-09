import json
import os
import logging
import re
from collections import defaultdict
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional

from dotenv import load_dotenv
from openai import OpenAI

from models import (
    AnalyzeResponse,
    ChatStats,
    ParticipantProfile,
    ParticipantStats,
    Recommendation,
    RelationshipSummary,
)
from telegram_parser import TelegramMessage, parse_telegram_html
from whatsapp_parser import parse_whatsapp_txt

load_dotenv()
logger = logging.getLogger(__name__)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1",
)
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Переключатель: используем ли реальный LLM или заглушку
USE_LLM = os.getenv("USE_LLM", "0") == "1"

# Лимиты из конфига
LLM_MAX_CHARS = int(os.getenv("LLM_MAX_CHARS", "60000"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "900"))

SYSTEM_PROMPT = """
Ты профессиональный психолог-аналитик переписок, который очень понятно дает результат анализа.
Старайся делать ПОДРОБНЫЙ анализ и выводы.
Работаешь только с анонимизированными данными.

Всегда считаем, что есть минимум два участника: USER_1 и USER_2.
Массивы "participants", "relationship" и "recommendations" НЕ ДОЛЖЕНЫ быть пустыми.
Обязательно создай записи хотя бы для USER_1 и USER_2, даже если данных мало —
в таком случае явно отмечай, что выводы ограничены.

Отвечай СТРОГО в формате JSON (без текста до или после объекта, все поля в JSON должны быть ЗАПОЛНЕНЫ):

{
  "participants": [
    {
      "id": "USER_1",
      "display_name": "USER_1",
      "traits": {
        "extroversion": "низкая/средняя/высокая",
        "emotional_stability": "...",
        "other": "...",
      },
      "summary": "подробное живое описание понятным языком для обывателя, 20-40 предложений и описания стиля общения"
    }
  ],
  "relationship": {
    "description": "подробное описание динамики взаимоотношений не менее 12 предложений",
    "red_flags": ["..."],
    "green_flags": ["..."]
  },
  "recommendations": [
    { "title": "краткий заголовок", "text": "1-2 абзаца с конкретным советом" }
  ]
}
"""

# ---- Шум / служебные сообщения ----

MEDIA_PLACEHOLDERS = {
    "not included, change data exporting settings to download.",
    "[media omitted]",
    "<media omitted>",
    "‎image omitted",
    "‎image omitted.",
    "изображение не сохранено",
    "медиафайл отсутствует",
    "Без медиафайлов",
    "Пропущенный аудиозвонок",
}

SYSTEM_PATTERNS = [
    r"сообщения и звонки .* защищены сквозным шифрованием",
    r"вы создали группу",
    r"вы изменили фото группы",
    r"вы изменили тему беседы",
    r"вы закрепили сообщение",
    r"вы удалили сообщение",
    r"Без медиафайлов",
    r"Пропущенный аудиозвонок",
]

_system_regexes = [re.compile(pat, re.IGNORECASE) for pat in SYSTEM_PATTERNS]


def _is_noise_text(text: str) -> bool:
    """
    Определяем, является ли текст "шумом":
    - пустые строки
    - стандартные заглушки медиа
    - системные сообщения (создание/изменение группы и т.п.)

    ВАЖНО: emoji, короткие реплики, «👍», «ок» и т.п. — НЕ считаем шумом.
    """
    if not text:
        return True

    stripped = text.strip()
    if not stripped:
        return True

    low = stripped.lower()

    # заглушки медиа
    if low in MEDIA_PLACEHOLDERS:
        return True

    # системные сообщения (по регуляркам)
    for rx in _system_regexes:
        if rx.search(low):
            return True

    return False


def _filter_noise_messages(messages: List[TelegramMessage]) -> List[TelegramMessage]:
    """
    Убираем из списка сообщений мусор:
    - чистые media-заглушки
    - системные сообщения
    Эмодзи и короткие тексты НЕ трогаем.
    """
    before = len(messages)
    cleaned: List[TelegramMessage] = []

    for msg in messages:
        txt = msg.text or ""
        if _is_noise_text(txt):
            continue
        cleaned.append(msg)

    removed = before - len(cleaned)
    logger.info(
        "[noise_filter] before=%d, after=%d, removed=%d",
        before,
        len(cleaned),
        removed,
    )
    return cleaned


def _extract_json_block(content: str) -> str:
    """На всякий случай вырезаем JSON-объект { ... } из произвольного текста."""
    if not content:
        return content
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return content[start : end + 1]
    return content


def _compute_stats_from_messages(messages: List[TelegramMessage]) -> ChatStats:
    total = len(messages)

    per_user_length: Dict[str, List[int]] = defaultdict(list)
    dates: List[datetime] = []

    for msg in messages:
        per_user_length[msg.from_name].append(len(msg.text))
        if msg.date:
            dates.append(msg.date)

    participants_stats: List[ParticipantStats] = []
    for user, lengths in per_user_length.items():
        # Берём только настоящих участников вида USER_1, USER_2, ...
        if not user.startswith("USER_"):
            continue

        count = len(lengths)
        avg_len = sum(lengths) / count if count else 0
        participants_stats.append(
            ParticipantStats(
                id=user,
                messages_count=count,
                avg_message_length=round(avg_len, 1),
            )
        )

    participants_stats.sort(key=lambda p: p.messages_count, reverse=True)

    first_dt = min(dates) if dates else None
    last_dt = max(dates) if dates else None

    return ChatStats(
        total_messages=total,
        participants=participants_stats,
        first_message_at=first_dt,
        last_message_at=last_dt,
    )


def _compute_stats_from_plain_text(text: str) -> ChatStats:
    """Простейшая статистика для нераспознанного формата."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return ChatStats(
        total_messages=len(lines),
        participants=[],
        first_message_at=None,
        last_message_at=None,
    )


def _filter_messages_by_date(
    messages: List[TelegramMessage],
    from_date: Optional[date],
    to_date: Optional[date],
) -> List[TelegramMessage]:
    """
    Фильтруем сообщения по диапазону дат (включительно).
    Если фильтрация дала пустой результат — возвращаем исходный список.
    """
    if not from_date and not to_date:
        return messages

    filtered: List[TelegramMessage] = []
    for msg in messages:
        if not msg.date:
            continue
        d = msg.date.date()
        if from_date and d < from_date:
            continue
        if to_date and d > to_date:
            continue
        filtered.append(msg)

    if not filtered:
        logger.warning(
            "После фильтрации по дате сообщений не осталось, "
            "используем полный набор сообщений.",
        )
        return messages

    return filtered


def _build_conversation_snippet(
    messages: List[TelegramMessage],
    max_chars: int = None,
    allowed_ids: Optional[set[str]] = None,
) -> str:
    """
    Готовим компактный текст для LLM: "USER_1: сообщение".
    Если переданы allowed_ids — берём только сообщения этих участников.
    """
    if max_chars is None:
        max_chars = LLM_MAX_CHARS
    lines: List[str] = []
    total_len = 0

    for msg in messages:
        if not msg.text.strip():
            continue

        if allowed_ids is not None and msg.from_name not in allowed_ids:
            continue

        line = f"{msg.from_name}: {msg.text}"
        if total_len + len(line) > max_chars:
            break

        lines.append(line)
        total_len += len(line)

    # fallback, если после фильтра по allowed_ids ничего не набрали
    if not lines:
        total_len = 0
        for msg in messages:
            if not msg.text.strip():
                continue
            line = f"{msg.from_name}: {msg.text}"
            if total_len + len(line) > max_chars:
                break
            lines.append(line)
            total_len += len(line)

    return "\n".join(lines)


def _build_plain_snippet(text: str, max_chars: int = None) -> str:
    if max_chars is None:
        max_chars = LLM_MAX_CHARS
    lines = [ln for ln in text.splitlines() if ln.strip()]
    snippet = "\n".join(lines)
    return snippet[:max_chars]


def _call_llm(
    conversation_snippet: str,
) -> Tuple[List[ParticipantProfile], RelationshipSummary, List[Recommendation]]:
    user_prompt = (
        "Ниже — анонимизированная переписка (диалог между 2 участниками USER_1 и USER_2).\n"
        "Твоя задача — на основе стиля сообщений и эмоциональных реакций:\n"
        "1) Составить развёрнутый психологический портрет каждого участника.\n"
        "2) Описать динамику их отношений.\n"
        "3) Дать практические рекомендации по улучшению общения.\n\n"
        "Используй только информацию из переписки, ничего не придумывай сверх наблюдаемого.\n\n"
        "ПЕРЕПИСКА:\n"
        + conversation_snippet
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=LLM_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        logger.exception("OpenAI API call failed: %r", exc)
        raise RuntimeError(f"Не удалось получить ответ от модели: {exc}") from exc

    content = completion.choices[0].message.content
    logger.info("LLM raw content: %r", content)
    logger.info("completion.usage: %r", getattr(completion, "usage", None))

    if not content or not content.strip():
        raise RuntimeError("Модель вернула пустой ответ")

    cleaned = _extract_json_block(content)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.exception("JSON decode error from LLM, cleaned=%r", cleaned)
        raise RuntimeError(f"Модель вернула невалидный JSON: {exc}") from exc

    participants: List[ParticipantProfile] = []
    for p in data.get("participants", []):
        participants.append(
            ParticipantProfile(
                id=p.get("id") or p.get("display_name") or "USER",
                display_name=p.get("display_name") or p.get("id") or "USER",
                traits=p.get("traits", {}),
                summary=p.get("summary", ""),
            )
        )

    if not participants:
        logger.error("LLM вернул пустой список participants, data=%r", data)
        raise RuntimeError("LLM вернул пустой список participants")

    rel_raw = data.get("relationship", {}) or {}
    relationship = RelationshipSummary(
        description=rel_raw.get("description", ""),
        red_flags=rel_raw.get("red_flags", []) or [],
        green_flags=rel_raw.get("green_flags", []) or [],
    )

    recommendations: List[Recommendation] = []
    for r in data.get("recommendations", []):
        recommendations.append(
            Recommendation(
                title=r.get("title", "Рекомендация"),
                text=r.get("text", ""),
            )
        )

    return participants, relationship, recommendations


def _build_dummy_response() -> Tuple[List[ParticipantProfile], RelationshipSummary, List[Recommendation]]:
    """Заглушка на время разработки."""
    dummy_participants = [
        ParticipantProfile(
            id="USER_1",
            display_name="USER_1",
            traits={
                "extroversion": "средняя",
                "emotional_stability": "средняя",
                "agreeableness": "высокая",
            },
            summary="Спокойный, в целом доброжелательный собеседник, склонен сглаживать конфликты.",
        ),
        ParticipantProfile(
            id="USER_2",
            display_name="USER_2",
            traits={
                "extroversion": "выше средней",
                "emotional_stability": "пониженная",
                "assertiveness": "высокая",
            },
            summary="Эмоциональный, инициативный, иногда может давить на своём мнении.",
        ),
    ]

    dummy_relationship = RelationshipSummary(
        description="Отношения в целом тёплые, но присутствуют эпизоды напряжения из-за разницы в стиле общения.",
        red_flags=[
            "Иногда игнорируются потребности одного из участников.",
            "Есть тенденция к пассивной агрессии в переписке.",
        ],
        green_flags=[
            "Есть юмор и поддержка.",
            "Обе стороны возвращаются к общению после конфликтов, что говорит о значимости связи.",
        ],
    )

    dummy_recommendations = [
        Recommendation(
            title="Проговаривать ожидания",
            text="Попробуйте явно говорить, чего вы ожидаете друг от друга, вместо пассивных намёков.",
        ),
        Recommendation(
            title="Фиксировать сложные темы",
            text="Сложные разговоры лучше выносить в отдельный диалог или голос, а не решать их поздно ночью в мессенджере.",
        ),
        Recommendation(
            title="Больше позитивного подкрепления",
            text="Замечайте и проговаривайте то, что вам нравится в поведении друг друга – это снижает общий фон напряжения.",
        ),
    ]

    return dummy_participants, dummy_relationship, dummy_recommendations


def compute_chat_stats_only(chat_text: str) -> ChatStats:
    """
    Лёгкий подсчёт статистики без LLM.
    Используется в /chat_meta.
    Здесь тоже режем шум, чтобы цифры были про «живые» сообщения.
    """
    is_html = "<html" in chat_text[:500].lower()

    if is_html:
        messages = parse_telegram_html(chat_text)
        if messages:
            cleaned = _filter_noise_messages(messages)
            return _compute_stats_from_messages(cleaned)
        else:
            return _compute_stats_from_plain_text(chat_text)
    else:
        wa_messages = parse_whatsapp_txt(chat_text)
        if wa_messages:
            cleaned = _filter_noise_messages(wa_messages)
            return _compute_stats_from_messages(cleaned)
        else:
            return _compute_stats_from_plain_text(chat_text)


def analyze_chat_text(
    chat_text: str,
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
) -> AnalyzeResponse:
    """
    Главная функция анализа.
    1) Определяет формат (Telegram HTML / WhatsApp txt / прочее)
    2) Опционально фильтрует сообщения по дате
    3) Режет шум (сервисные/медиа-заглушки), но сохраняет эмодзи и короткие реплики
    4) Считает статистику
    5) Либо дергает LLM, либо возвращает заглушку
    """
    is_html = "<html" in chat_text[:500].lower()

    if is_html:
        messages = parse_telegram_html(chat_text)
        if messages:
            # сначала по дате
            by_date = _filter_messages_by_date(messages, from_date, to_date)
            # затем вычищаем шум
            filtered = _filter_noise_messages(by_date)

            stats = _compute_stats_from_messages(filtered)
            main_ids = {p.id for p in stats.participants} or None
            snippet = _build_conversation_snippet(
                filtered,
                allowed_ids=main_ids,
            )
        else:
            stats = _compute_stats_from_plain_text(chat_text)
            snippet = _build_plain_snippet(chat_text)
    else:
        wa_messages = parse_whatsapp_txt(chat_text)
        if wa_messages:
            by_date = _filter_messages_by_date(wa_messages, from_date, to_date)
            filtered = _filter_noise_messages(by_date)
            stats = _compute_stats_from_messages(filtered)
            main_ids = {p.id for p in stats.participants} or None
            snippet = _build_conversation_snippet(
                filtered,
                allowed_ids=main_ids,
            )
        else:
            stats = _compute_stats_from_plain_text(chat_text)
            snippet = _build_plain_snippet(chat_text)

    print(
        f"[analyze_chat_text] format={'html' if is_html else 'txt'}, "
        f"snippet_len={len(snippet)}, total_messages={stats.total_messages}"
    )

    logger.info(
        "[analyze_chat_text] format=%s, snippet_len=%d, total_messages=%d",
        "html" if is_html else "txt",
        len(snippet),
        stats.total_messages,
    )

    if USE_LLM:
        try:
            participants, relationship, recommendations = _call_llm(snippet)
        except Exception:
            logger.exception("LLM call failed, используем заглушку")
            participants, relationship, recommendations = _build_dummy_response()
    else:
        participants, relationship, recommendations = _build_dummy_response()

    return AnalyzeResponse(
        participants=participants,
        relationship=relationship,
        recommendations=recommendations,
        stats=stats,
    )
