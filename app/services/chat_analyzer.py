"""Chat analysis service using LLM."""
import json
import logging
from datetime import date
from typing import List, Optional, Tuple

from openai import OpenAI

from .chat_parser import parse_chat_text
from .telegram_parser import TelegramMessage
from ..config import settings
from ..models.schemas import (
    AnalyzeResponse,
    ParticipantProfile,
    Recommendation,
    RelationshipSummary,
)

logger = logging.getLogger(__name__)

# OpenAI client
client = OpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url,
)

SYSTEM_PROMPT = """
Ты профессиональный психолог-аналитик переписок, который очень понятно дает результат анализа.
Старайся делать ПОДРОБНЫЙ анализ и выводы.
Работаешь только с анонимизированными данными.

Важно: личные данные (имена, фамилии, точные адреса, номера телефонов, e-mail, ссылки на профили и любые уникальные идентификаторы) НЕ ДОЛЖНЫ появляться в ответе.
Не упоминай реальные имена людей, компаний или сервисов, даже если они есть в переписке.
Используй обобщённые обозначения (например, "подруга", "коллега", "родственник"), а названия городов указывать можно.

Формат ответа КРИТИЧЕН: если JSON будет невалидным, весь результат анализа будет отброшен и заменён технической заглушкой.

Всегда определяй количество участников на основе переписки.
Массивы "participants", "relationship" и "recommendations" НЕ ДОЛЖНЫ быть пустыми.
Обязательно создай записи для КАЖДОГО уникального участника (USER_1, USER_2, USER_3 и т.д.), который присутствует в переписке.
Если данных для конкретного участника мало — явно отмечай, что выводы ограничены.

ОЧЕНЬ ВАЖНО:
- ОТВЕЧАЙ СТРОГО ТОЛЬКО ОДНИМ JSON-ОБЪЕКТОМ.
- НЕ ДОБАВЛЯЙ никакого текста ДО или ПОСЛЕ JSON.
- НЕ ИСПОЛЬЗУЙ форматирование Markdown (никаких ```json, ``` или других тегов).
- НЕ ДОБАВЛЯЙ разделы вроде "Дополнительный анализ", комментарии, пояснения и т.п.
- В ответе ДОЛЖЕН быть только один объект { ... } без обёртки и лишних символов.

Структура ответа (ВСЕ ПОЛЯ ОБЯЗАТЕЛЬНЫ):

{
  "participants": [
    {
      "id": "USER_1",
      "display_name": "USER_1",
      "traits": {
        "extroversion": "низкая/средняя/высокая",
        "emotional_stability": "...",
        "other": "..."
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

ЕЩЁ РАЗ: никаких пояснений, Markdown-разметки, комментариев или дополнительного текста — ТОЛЬКО JSON-объект, начинающийся с '{' и заканчивающийся '}'.
"""


def _extract_json_block(content: str) -> str:
    """На всякий случай вырезаем JSON-объект { ... } из произвольного текста."""
    if not content:
        return content
    start = content.find("{")
    end = content.rfind("}")

    # Если модель добавила markdown-обёртку ```json ... ``` или разделы вроде "Дополнительный анализ",
    # стараемся обрезать только чистый JSON-объект.
    # Находим позицию возможных маркеров конца полезного JSON.
    cut_markers = []
    for marker in ["```", "Дополнительный анализ", "Дополнительный анализ", "---"]:
        idx = content.find(marker, end + 1)
        if idx != -1:
            cut_markers.append(idx)
    if cut_markers:
        hard_end = min(cut_markers)
        # Ищем последнюю '}' перед первым маркером
        end_before_marker = content.rfind("}", start, hard_end)
        if end_before_marker != -1 and end_before_marker > start:
            end = end_before_marker

    if start != -1 and end != -1 and end > start:
        return content[start : end + 1]
    return content


def _build_conversation_snippet(
    messages: List[TelegramMessage],
    max_chars: Optional[int] = None,
    allowed_ids: Optional[set[str]] = None,
) -> str:
    """
    Готовим компактный текст для LLM: "USER_1: сообщение".
    Если переданы allowed_ids — берём только сообщения этих участников.
    """
    if max_chars is None:
        max_chars = settings.llm_max_chars

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


def _build_plain_snippet(text: str, max_chars: Optional[int] = None) -> str:
    """Для нераспознанного формата."""
    if max_chars is None:
        max_chars = settings.llm_max_chars
    lines = [ln for ln in text.splitlines() if ln.strip()]
    snippet = "\n".join(lines)
    return snippet[:max_chars]


def _call_llm(
    conversation_snippet: str,
) -> Tuple[
    List[ParticipantProfile],
    RelationshipSummary,
    List[Recommendation],
    Optional[dict],  # token usage info
]:
    """Вызывает LLM для анализа и возвращает результат + usage."""
    user_prompt = (
        "Ниже — анонимизированная переписка между участниками (USER_1, USER_2, и возможно другими).\n"
        "Твоя задача — на основе стиля сообщений и эмоциональных реакций:\n"
        "1) Составить развёрнутый психологический портрет КАЖДОГО уникального участника, который присутствует в переписке.\n"
        "2) Описать динамику их отношений.\n"
        "3) Дать практические рекомендации по улучшению общения.\n\n"
        "Используй только информацию из переписки, ничего не придумывай сверх наблюдаемого.\n\n"
        "ПЕРЕПИСКА:\n" + conversation_snippet
    )

    try:
        completion = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=settings.llm_max_tokens,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        logger.exception("OpenAI API call failed: %r", exc)
        raise RuntimeError(f"Не удалось получить ответ от модели: {exc}") from exc

    content = completion.choices[0].message.content
    usage = getattr(completion, "usage", None)
    
    # Логируем usage для статистики
    if usage:
        logger.info(
            "LLM usage: prompt_tokens=%d, completion_tokens=%d, total_tokens=%d",
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
        )
        usage_dict = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
    else:
        usage_dict = None

    logger.info("LLM raw content: %r", content)

    if not content or not content.strip():
        raise RuntimeError("Модель вернула пустой ответ")

    cleaned = _extract_json_block(content)

    # Попытка "подлечить" JSON-ответ: обрезаем по последней закрывающей скобке
    last_brace = cleaned.rfind("}")
    if last_brace != -1:
        cleaned = cleaned[: last_brace + 1]

    # Если модель оборвала ответ - пытаемся закрыть структуру
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse failed, attempting repair: %s", exc)
        # Проверяем, есть ли все обязательные поля в ответе
        has_participants = '"participants"' in cleaned
        has_relationship = '"relationship"' in cleaned
        has_recommendations = '"recommendations"' in cleaned
        
        if has_participants and not (has_relationship and has_recommendations):
            # Модель оборвала ответ - дописываем недостающие части
            repaired = cleaned.rstrip()
            
            # Убираем незакрытые кавычки/запятые в конце
            while repaired and repaired[-1] in ('"', ',', ' ', '\n', '\r', '\t'):
                if repaired[-1] == '"':
                    # Если есть незакрытая кавычка - убираем её
                    repaired = repaired[:-1].rstrip()
                    break
                repaired = repaired[:-1]
            
            # Проверяем, есть ли незакрытое поле "summary" в последнем участнике
            # Ищем последнее вхождение "summary": "
            last_summary_start = repaired.rfind('"summary": "')
            if last_summary_start != -1:
                # Проверяем, закрыто ли это поле (есть ли закрывающая кавычка после значения)
                after_summary = repaired[last_summary_start + len('"summary": "'):]
                # Считаем кавычки с учётом экранирования
                # Простая проверка: если нет закрывающей кавычки или она не закрыта корректно
                if not after_summary or after_summary.count('"') == 0:
                    # summary оборван - добавляем закрывающую кавычку
                    repaired += '"'
            
            # Закрываем все открытые структуры:
            # 1. Проверяем, есть ли все обязательные поля в последнем участнике
            # Ищем последнее вхождение "traits"
            last_traits = repaired.rfind('"traits"')
            last_summary = repaired.rfind('"summary"')
            
            # Если traits есть, но summary после него нет - добавляем пустой summary
            if last_traits != -1 and (last_summary == -1 or last_summary < last_traits):
                # Нужно добавить summary перед закрытием объекта участника
                # Ищем закрывающую скобку traits
                traits_end = repaired.find('}', last_traits)
                if traits_end != -1:
                    # Добавляем запятую после traits и summary
                    repaired += ',"summary":""'
            
            # 2. Закрываем последний объект участника (если он не закрыт)
            if not repaired.endswith('}'):
                repaired += '}'
            
            # 3. Закрываем массив participants (если он не закрыт)
            if not ']' in repaired[repaired.rfind('"participants"'):] or repaired.count('[') > repaired.count(']'):
                repaired += ']'
            
            # 4. Добавляем недостающие поля
            if not has_relationship:
                repaired += ',"relationship":{"description":"","red_flags":[],"green_flags":[]}'
            if not has_recommendations:
                repaired += ',"recommendations":[]'
            
            # 5. Закрываем корневой объект
            if not repaired.endswith('}'):
                repaired += '}'
            
            try:
                data = json.loads(repaired)
                logger.info("JSON successfully repaired")
            except json.JSONDecodeError as repair_exc:
                logger.exception("JSON repair failed, original cleaned=%r, repaired=%r", cleaned, repaired)
                raise RuntimeError(f"Модель вернула невалидный JSON: {exc}") from exc
        else:
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

    return participants, relationship, recommendations, usage_dict


def _build_dummy_response() -> Tuple[
    List[ParticipantProfile], RelationshipSummary, List[Recommendation], None
]:
    """Заглушка на время разработки."""
    fallback_text = "Данный анализ не является действительным результатом работы сервиса."
    
    dummy_participants = [
        ParticipantProfile(
            id="USER_1",
            display_name="USER_1",
            traits={
                "extroversion": fallback_text,
                "emotional_stability": fallback_text,
                "agreeableness": fallback_text,
            },
            summary=fallback_text,
        ),
        ParticipantProfile(
            id="USER_2",
            display_name="USER_2",
            traits={
                "extroversion": fallback_text,
                "emotional_stability": fallback_text,
                "assertiveness": fallback_text,
            },
            summary=fallback_text,
        ),
    ]

    dummy_relationship = RelationshipSummary(
        description=fallback_text,
        red_flags=[fallback_text],
        green_flags=[fallback_text],
    )

    dummy_recommendations = [
        Recommendation(
            title=fallback_text,
            text=fallback_text,
        ),
        Recommendation(
            title=fallback_text,
            text=fallback_text,
        ),
        Recommendation(
            title=fallback_text,
            text=fallback_text,
        ),
    ]

    return dummy_participants, dummy_relationship, dummy_recommendations, None


def analyze_chat_text(
    chat_text: str,
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
) -> Tuple[AnalyzeResponse, Optional[dict]]:
    """
    Главная функция анализа.
    Возвращает (AnalyzeResponse, token_usage_dict).
    """
    messages, stats = parse_chat_text(chat_text, from_date, to_date)

    if messages:
        main_ids = {p.id for p in stats.participants} or None
        snippet = _build_conversation_snippet(messages, allowed_ids=main_ids)
    else:
        snippet = _build_plain_snippet(chat_text)

    logger.info(
        "[analyze_chat_text] snippet_len=%d, total_messages=%d",
        len(snippet),
        stats.total_messages,
    )

    token_usage = None
    is_fallback = False
    error_message = None
    
    if settings.use_llm:
        try:
            participants, relationship, recommendations, token_usage = _call_llm(snippet)
        except Exception as exc:
            logger.exception("LLM call failed, используем заглушку")
            participants, relationship, recommendations, _ = _build_dummy_response()
            is_fallback = True
            error_message = f"{type(exc).__name__}: {str(exc)}"
    else:
        participants, relationship, recommendations, _ = _build_dummy_response()

    response = AnalyzeResponse(
        participants=participants,
        relationship=relationship,
        recommendations=recommendations,
        stats=stats,
        is_fallback=is_fallback,
        error_message=error_message,
    )

    return response, token_usage
