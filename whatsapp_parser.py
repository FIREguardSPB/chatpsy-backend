from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional, Tuple

from dateutil import parser as dateparser

from telegram_parser import TelegramMessage

# Форматы строк WhatsApp:
# 1) [07/04/2018, 14:11:22] Mike: Hi
# 2) 9/5/21, 8:07 PM - Me: Lol romantic comedy
# 3) 09.01.2023, 19:58 - USER_1: Привет
# и т.п.

DATE_PATTERN = r"\d{1,2}[./]\d{1,2}[./]\d{2,4}"
TIME_PATTERN = r"\d{1,2}:\d{2}(?::\d{2})?(?:\s?(?:AM|PM|am|pm))?"

WHATSAPP_PATTERNS = [
    # [dd/mm/yyyy, hh:mm(:ss) AM] Name: Msg
    re.compile(
        rf"""^\[
        (?P<date>{DATE_PATTERN}),
        \s*
        (?P<time>{TIME_PATTERN})
        \]
        \s*
        (?P<name>[^:]+)
        :
        \s*
        (?P<msg>.*)
        $""",
        re.VERBOSE,
    ),
    # dd/mm/yyyy, hh:mm(:ss) AM - Name: Msg
    re.compile(
        rf"""^
        (?P<date>{DATE_PATTERN}),
        \s*
        (?P<time>{TIME_PATTERN})
        \s*[-–]\s*
        (?P<name>[^:]+)
        :
        \s*
        (?P<msg>.*)
        $""",
        re.VERBOSE,
    ),
]

# Типичные сервисные/системные строки, которые ломают формат
SYSTEM_PREFIXES = (
    "Messages and calls are end-to-end encrypted",
    "Сообщения и звонки защищены сквозным шифрованием",
)

MEDIA_MARKERS = (
    "<Media omitted>",
    "Media omitted",
    "<Медиа изъято>",
    "изображение скрыто",
    "image omitted",
    "video omitted",
)


def _match_whatsapp_header(line: str) -> Optional[Tuple[Optional[datetime], str, str]]:
    """
    Пытается распарсить строку как заголовок сообщения WhatsApp.
    Возвращает (datetime | None, name, msg) или None.
    """
    for pattern in WHATSAPP_PATTERNS:
        m = pattern.match(line)
        if m:
            date_str = m.group("date")
            time_str = m.group("time")
            name = m.group("name").strip()
            msg = m.group("msg").strip()

            try:
                dt = dateparser.parse(f"{date_str} {time_str}", dayfirst=True)
            except Exception:
                dt = None

            return dt, name, msg
    return None


def parse_whatsapp_txt(text: str) -> List[TelegramMessage]:
    """
    Парсит txt-экспорт WhatsApp в список TelegramMessage
    (from_name, text, date). Поддерживает основные варианты формата,
    многострочные сообщения и "экзотику" вроде <Media omitted>.
    """
    messages: List[TelegramMessage] = []
    current: Optional[TelegramMessage] = None

    for raw_line in text.splitlines():
        line = raw_line.strip("\r\n")
        if not line.strip():
            continue

        # Системные строки (про шифрование и т.п.) просто скипаем
        if any(line.startswith(prefix) for prefix in SYSTEM_PREFIXES):
            if current is not None:
                messages.append(current)
                current = None
            continue

        header = _match_whatsapp_header(line)
        if header:
            dt, name, msg = header

            # закрываем предыдущее сообщение
            if current is not None:
                messages.append(current)

            # Если сообщение — чистый маркер медиа, текст не засоряем
            if msg in MEDIA_MARKERS:
                msg = ""

            current = TelegramMessage(
                from_name=name,
                text=msg,
                date=dt,
            )
        else:
            # продолжение предыдущего сообщения
            if current is not None:
                if line in MEDIA_MARKERS:
                    # <Media omitted> как отдельной строкой — тоже не добавляем
                    continue
                current.text += "\n" + line.strip()
            # если current нет — странная строка, просто пропускаем

    if current is not None:
        messages.append(current)

    return messages
