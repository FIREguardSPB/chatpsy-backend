"""Chat analysis service using LLM."""
import json
import logging
from datetime import date
from typing import List, Optional, Tuple, Set

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


def model_supports_json_format(model_name: str) -> bool:
    """
    Определяем, поддерживает ли модель строгий JSON-вывод через response_format={"type": "json_object"}.
    Ориентируемся на GPT-линейку. Для остальных (Claude, локальные, open-models) вернём False.
    """
    if not model_name:
        return False

    name = model_name.lower()
    # Можно расширять список по мере необходимости
    return (
        name.startswith("gpt-")
        or "gpt-4o" in name
        or "gpt-4.1" in name
        or "gpt-4o-mini" in name
    )


SYSTEM_PROMPT = """
Ты — профессиональный психолог-аналитик переписок и СТРОГИЙ ГЕНЕРАТОР JSON.

Твои задачи:
1) Проанализировать анонимизированную переписку (USER_1, USER_2, ...).
2) Составить психологический портрет каждого участника.
3) Описать динамику их отношений.
4) Выделить тревожные моменты (red_flags) и здоровые моменты (green_flags).
5) Дать практические рекомендации.

Анонимность:
- Все реальные имена, фамилии, ники и т.п. считаются чувствительными.
- В переписке могут встречаться реальные имена (например, «Максим», «Антон», «Катя» и т.п.).
- Тебе СТРОГО ЗАПРЕЩЕНО использовать эти имена в выводе.
- ВО ВСЕХ текстах (traits.other, summary, relationship.description, red_flags, green_flags, recommendations.text)
  ты можешь ссылаться на людей ТОЛЬКО как на USER_1, USER_2 и т.п., либо нейтрально «первый участник», «второй участник».
- НЕ ПИШИ конструкции вида «USER_1 (Максим)» или «Максим (USER_1)» — только идентификаторы USER_1, USER_2 и т.д.

Переписка уже частично анонимизирована:
- имена отправителей заменены на технические идентификаторы (USER_1, USER_2 и т.п.);
- однако внутри текстов сообщений могут оставаться имена — их НУЖНО ИГНОРИРОВАТЬ и НЕ ВОСПРОИЗВОДИТЬ в ответе.

ОЧЕНЬ ВАЖНО: ФОРМАТ ОТВЕТА

Ты ОБЯЗАН вернуть СТРОГО ВАЛИДНЫЙ JSON. Ничего больше.

Тебе КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО:
- использовать markdown или code fences (никаких ```json и т.п.);
- добавлять любой текст до JSON или после JSON;
- писать комментарии, пояснения, "вот ваш JSON", "итог", "анализ" и т.п.;
- менять структуру JSON;
- пропускать обязательные ключи;
- добавлять свои поля;
- переименовывать ключи.

СТРУКТУРА ДОЛЖНА БЫТЬ СТРОГО ТАКОЙ:

{
  "participants": [
    {
      "id": "USER_1",
      "display_name": "USER_1",
      "traits": {
        "extroversion": "краткое описание уровня общительности",
        "emotional_stability": "описание устойчивости к стрессу и перепадам настроения",
        "other": "любые дополнительные наблюдения по характеру, стилю общения, особенностям мышления и поведения (БЕЗ РЕАЛЬНЫХ ИМЁН)"
      },
      "summary": "подробный, живой, понятный текстовый портрет человека на основе переписки (БЕЗ РЕАЛЬНЫХ ИМЁН, только USER_1, USER_2 и т.п.)"
    }
  ],
  "relationship": {
    "description": "подробное описание динамики отношений между участниками, их ролей, конфликтов, поддержки, доверия и т.п. (БЕЗ РЕАЛЬНЫХ ИМЁН)", 
    "red_flags": [
      "каждый элемент — краткая формулировка настораживающего момента в отношениях или поведении (БЕЗ РЕАЛЬНЫХ ИМЁН)",
      "ещё один тревожный момент, если он есть"
    ],
    "green_flags": [
      "каждый элемент — краткая формулировка здорового, поддерживающего аспекта отношений или поведения (БЕЗ РЕАЛЬНЫХ ИМЁН)",
      "ещё один положительный момент, если он есть"
    ]
  },
  "recommendations": [
    {
      "title": "краткий заголовок рекомендации",
      "text": "подробное объяснение, что стоит изменить, какие шаги предпринять, как улучшить общение, на что обратить внимание (БЕЗ РЕАЛЬНЫХ ИМЁН)"
    }
  ]
}

ТРЕБОВАНИЯ К ПОДРОБНОСТИ:

- Для КАЖДОГО участника:
  - traits.other — 2–4 ключевые характеристики (через запятую), без имён;
  - summary — развёрнутый текст: не менее 35-40 предложений, с примерами типичных реакций и поведения, но без упоминания реальных имён.

- Для relationship.description:
  - не менее 10–15 предложений;
  - опиши роли участников, кто инициирует контакт, кто поддерживает, где напряжение;
  - приводи наблюдения по стилю сообщений (частота, объём, тон, эмодзи и т.д.), но без имён.

- Для red_flags / green_flags:
  - перечисли все значимые моменты;
  - каждый пункт — отдельная, понятная формулировка без имён.

- Для recommendations:
  - желательно 4–6 рекомендаций;
  - каждая recommendation.text — 3–6 предложений с конкретными шагами.

Не сокращай текст ради краткости. Стремись к глубокому, насыщенному описанию, но строго соблюдай анонимность (ТОЛЬКО USER_1, USER_2 и т.п.).

Правила JSON:
- ВСЕ строки в двойных кавычках.
- НЕТ лишних запятых.
- НЕТ лишних ключей.
- Пустые массивы допустимы, но ключи должны присутствовать всегда.
- null использовать только при крайней необходимости, лучше дать краткую строку.

Если данных мало, всё равно заполни ВСЮ структуру:
- participants — хотя бы один объект с честным описанием того, что данных немного;
- relationship — честно напиши, что информации мало, но верни description, red_flags, green_flags;
- recommendations — минимум одна рекомендация.

Если тебе хочется написать что-то вне JSON — НЕ ПИШИ ЭТО.
Просто верни корректный JSON по указанной структуре.
"""

def _repair_truncated_json(cleaned: str) -> Optional[dict]:
    """
    Пытается починить типичный кейс: модель выдала почти полный JSON,
    но в самом конце не дописала закрывающие ]} для массива recommendations и корневого объекта.

    Возвращает dict, если починить удалось, иначе None.
    """
    stripped = cleaned.rstrip()

    # Быстрый подсчёт скобок
    open_curly = stripped.count("{")
    close_curly = stripped.count("}")
    open_sq = stripped.count("[")
    close_sq = stripped.count("]")

    # Типичный наш кейс: не хватает ровно одной ] и одной }
    if (
        stripped.endswith("}")
        and open_curly - close_curly == 1
        and open_sq - close_sq == 1
    ):
        candidate = stripped + "\n  ]\n}"
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    return None
    

def _extract_json_block(content: str) -> str:
    """
    Вырезаем JSON-объект { ... } из ответа модели.
    Поддерживаем случаи, когда модель всё ещё умудрилась завернуть в ```json ... ```.
    """
    if not content:
        return content

    # 1. Если модель обернула всё в ```...```, вырезаем внутренность
    if "```" in content:
        first = content.find("```")
        last = content.rfind("```")
        if first != -1 and last != -1 and last > first:
            inner = content[first + 3 : last]
            content = inner

    # 2. Убираем возможный префикс "json"/"json\n"
    stripped = content.lstrip()
    if stripped.lower().startswith("json"):
        stripped = stripped[4:]
        content = stripped.lstrip()

    # 3. Берём от первой { до последней }
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return content[start : end + 1]

    # Если вообще не нашли фигурные скобки — возвращаем как есть
    return content


def _build_conversation_snippet(
    messages: List[TelegramMessage],
    max_chars: Optional[int] = None,
    allowed_ids: Optional[Set[str]] = None,
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
    if not lines and messages:
        for msg in messages:
            if not msg.text.strip():
                continue
            line = f"{msg.from_name}: {msg.text}"
            if total_len + len(line) > max_chars:
                break
            lines.append(line)
            total_len += len(line)

    return "\n".join(lines)


def _build_plain_snippet(chat_text: str, max_chars: Optional[int] = None) -> str:
    """Простейший fallback, если парсер не смог распознать структуру."""
    if max_chars is None:
        max_chars = settings.llm_max_chars

    return chat_text[:max_chars]


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
        "Ниже — анонимизированная переписка между участниками (USER_1, USER_2 и другими).\n"
        "Проанализируй её и на основе СТИЛЯ общения, эмоций, динамики, реакции участников:\n"
        "1) Составь психологический портрет КАЖДОГО участника.\n"
        "2) Опиши динамику их взаимоотношений.\n"
        "3) Выдели тревожные моменты (red_flags) и здоровые (green_flags).\n"
        "4) Дай практические, прикладные рекомендации.\n\n"
        "ОЧЕНЬ ВАЖНО: верни ТОЛЬКО валидный JSON строго по структуре из системной инструкции. "
        "Никакого текста до JSON и после JSON, никаких комментариев, markdown или пояснений.\n"
    )

    params = {
        "model": settings.openai_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_prompt},
            {"role": "user", "content": conversation_snippet},
        ],
        # как было раньше — температура 0 для максимально стабильного вывода
        "temperature": 0.5,
        "max_tokens": settings.llm_max_tokens,
    }

    if model_supports_json_format(settings.openai_model):
        # Для GPT-моделей просим строгий JSON-объект
        params["response_format"] = {"type": "json_object"}

    completion = client.chat.completions.create(**params)

    content = completion.choices[0].message.content
    usage = getattr(completion, "usage", None)
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

    # Даже если это GPT с json_object, content обычно всё равно строка с JSON
    cleaned = _extract_json_block(content)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.exception("JSON decode error from LLM, cleaned=%r", cleaned)

        # Пытаемся автоматически починить типичный случай усечённого конца (нет ]})
        repaired = _repair_truncated_json(cleaned)
        if repaired is not None:
            logger.warning("JSON от LLM был усечён, но успешно восстановлен автоматически (добавили завершающее ]})")
            data = repaired
        else:
            raise RuntimeError(f"Модель вернула невалидный JSON: {exc}") from exc

    # Если модель по какой-то причине вернула обёртку с полем is_fallback
    if isinstance(data, dict) and data.get("is_fallback"):
        raise RuntimeError(
            f"Модель вернула невалидный JSON: {data.get('error_message', 'Неизвестная ошибка')}"
        )

    participants: List[ParticipantProfile] = []
    for p in data.get("participants", []):
        participants.append(
            ParticipantProfile(
                id=p.get("id") or p.get("display_name") or "USER",
                display_name=p.get("display_name") or p.get("id") or "USER",
                traits=p.get("traits", {}) or {},
                summary=p.get("summary", "") or "",
            )
        )

    if not participants:
        logger.warning("LLM вернул пустой список participants, используем фолбэк")
        fallback_participants, fallback_relationship, fallback_recommendations, _ = _build_dummy_response()
        return (
            fallback_participants,
            fallback_relationship,
            fallback_recommendations,
            usage_dict,
        )

    rel_raw = data.get("relationship", {}) or {}
    relationship = RelationshipSummary(
        description=rel_raw.get("description", "") or "",
        red_flags=rel_raw.get("red_flags", []) or [],
        green_flags=rel_raw.get("green_flags", []) or [],
    )

    recommendations: List[Recommendation] = []
    for r in data.get("recommendations", []):
        recommendations.append(
            Recommendation(
                title=r.get("title", "Рекомендация") or "Рекомендация",
                text=r.get("text", "") or "",
            )
        )

    return participants, relationship, recommendations, usage_dict


def _build_dummy_response() -> Tuple[
    List[ParticipantProfile],
    RelationshipSummary,
    List[Recommendation],
    Optional[dict],
]:
    """
    Строит "заглушку" ответа на случай, если LLM не смог дать валидный JSON
    или случилась ошибка на стороне модели/сети.
    """
    fallback_text = (
        "К сожалению, произошла ошибка при обработке переписки. "
        "Попробуйте ещё раз чуть позже. Если ошибка повторяется, "
        "уменьшите объём переписки или сократите вложения."
    )

    dummy_participants = [
        ParticipantProfile(
            id="USER_1",
            display_name="USER_1",
            traits={
                "extroversion": fallback_text,
                "emotional_stability": fallback_text,
                "other": fallback_text,
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

    participants: List[ParticipantProfile] = []
    relationship: RelationshipSummary
    recommendations: List[Recommendation] = []
    token_usage: Optional[dict] = None

    is_fallback = False
    error_message = None

    if snippet.strip():
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

