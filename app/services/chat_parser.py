"""Chat parsing and preprocessing service."""
import logging
import re
from collections import defaultdict
from datetime import date, datetime
from typing import Dict, List, Optional

from .telegram_parser import TelegramMessage, parse_telegram_html
from .whatsapp_parser import parse_whatsapp_txt
from ..models.schemas import ChatStats, ParticipantStats

logger = logging.getLogger(__name__)

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


def filter_noise_messages(messages: List[TelegramMessage]) -> List[TelegramMessage]:
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


def filter_messages_by_date(
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


def compute_stats_from_messages(messages: List[TelegramMessage]) -> ChatStats:
    """Вычисляет статистику из списка сообщений."""
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


def compute_stats_from_plain_text(text: str) -> ChatStats:
    """Простейшая статистика для нераспознанного формата."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return ChatStats(
        total_messages=len(lines),
        participants=[],
        first_message_at=None,
        last_message_at=None,
    )


def parse_chat_text(
    chat_text: str,
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
) -> tuple[List[TelegramMessage], ChatStats]:
    """
    Парсит текст чата, фильтрует по дате и шуму, возвращает сообщения и статистику.
    """
    is_html = "<html" in chat_text[:500].lower()

    if is_html:
        messages = parse_telegram_html(chat_text)
        if messages:
            by_date = filter_messages_by_date(messages, from_date, to_date)
            filtered = filter_noise_messages(by_date)
            stats = compute_stats_from_messages(filtered)
            return filtered, stats
        else:
            stats = compute_stats_from_plain_text(chat_text)
            return [], stats
    else:
        wa_messages = parse_whatsapp_txt(chat_text)
        if wa_messages:
            by_date = filter_messages_by_date(wa_messages, from_date, to_date)
            filtered = filter_noise_messages(by_date)
            stats = compute_stats_from_messages(filtered)
            return filtered, stats
        else:
            stats = compute_stats_from_plain_text(chat_text)
            return [], stats
