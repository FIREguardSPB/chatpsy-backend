"""Feedback route."""
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request

from ...config import settings
from ...core.rate_limiter import rate_limiter
from ...dependencies import get_client_ip
from ...models.schemas import FeedbackRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/feedback")
async def send_feedback(
    body: FeedbackRequest,
    request: Request,
    client_ip: str = Depends(get_client_ip),
):
    """
    Принимаем отзыв от пользователя, сохраняем в файл и
    начисляем бонусные анализы для его IP (один раз).
    """
    now = datetime.now(timezone.utc).isoformat()

    # 1) пишем отзыв в отдельный лог
    try:
        settings.log_dir.mkdir(exist_ok=True)
        feedback_path = settings.log_dir / "feedback.log"
        line = json.dumps(
            {
                "ts": now,
                "ip": client_ip,
                "text": body.text,
                "contact": body.contact,
            },
            ensure_ascii=False,
        )
        with open(feedback_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        # логируем, но пользователю всё равно возвращаем 200, чтобы не бесить
        logger.exception("Не удалось записать отзыв в файл: %r", exc)

    # 2) докидываем лимит для этого IP (если ещё не получал бонус)
    granted = rate_limiter.grant_feedback_bonus(client_ip)
    ip_stats = rate_limiter.get_ip_stats(client_ip)

    logger.info(
        "Feedback от IP %s, начислено %d анализов (limit %d, used %d)",
        client_ip,
        granted,
        ip_stats["analyze_limit"],
        ip_stats["analyze_used"],
    )

    return {
        "ok": True,
        "granted": granted,
        "new_limit": ip_stats["analyze_limit"],
        "used": ip_stats["analyze_used"],
    }
