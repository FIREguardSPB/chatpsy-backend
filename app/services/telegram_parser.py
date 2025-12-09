from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from bs4 import BeautifulSoup
from dateutil import parser as dateparser


@dataclass
class TelegramMessage:
    from_name: str
    text: str
    date: Optional[datetime]


def parse_telegram_html(html_str: str) -> List[TelegramMessage]:
    """
    Парсит HTML-экспорт Telegram и отдаёт список сообщений.
    Учитывает сообщения с class="message ..." + "joined" и т.п.
    """

    soup = BeautifulSoup(html_str, "html.parser")
    messages: List[TelegramMessage] = []

    last_from_name: Optional[str] = None

    for msg_div in soup.select("div.message"):
        body = msg_div.find("div", class_="body")
        if not body:
            continue

        # автор
        from_div = body.find("div", class_="from_name", recursive=False)
        if from_div is not None:
            from_name = from_div.get_text(strip=True)
            last_from_name = from_name
        else:
            # joined сообщение — автор тот же, что и у предыдущего
            if last_from_name is None:
                continue
            from_name = last_from_name

        # текст
        text_div = body.find("div", class_="text")
        if text_div is not None:
            text = text_div.get_text(separator="\n", strip=True)
        else:
            text = ""

        # время
        date_div = body.select_one("div.pull_right.date.details")
        msg_dt: Optional[datetime] = None
        if date_div is not None:
            raw = date_div.get("title") or date_div.get_text(strip=True)
            try:
                msg_dt = dateparser.parse(raw)
            except Exception:
                msg_dt = None

        messages.append(TelegramMessage(from_name=from_name, text=text, date=msg_dt))

    return messages
