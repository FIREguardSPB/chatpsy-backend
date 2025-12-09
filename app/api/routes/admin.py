"""Admin routes."""
import logging

from fastapi import APIRouter, HTTPException

from ...config import settings
from ...core.rate_limiter import rate_limiter

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/add_credits")
def admin_add_credits(token: str, ip: str, n: int = 1):
    """
    Админ-эндпоинт с авторизацией:
    /admin/add_credits?token=SECRET&ip=1.2.3.4&n=3
    Добавляет n дополнительных анализов указанному IP.
    """
    if not settings.admin_token or token != settings.admin_token:
        logger.warning("Unauthorized admin access attempt")
        raise HTTPException(status_code=403, detail="Forbidden")

    if n <= 0:
        raise HTTPException(status_code=400, detail="n должно быть > 0")

    result = rate_limiter.add_credits(ip, n)
    return result

@router.post("/set_limit")
def admin_set_limit(token: str, ip: str, limit: int):
    """
    Устанавливает абсолютный лимит анализов указанному IP.
    /admin/set_limit?token=SECRET&ip=1.2.3.4&limit=3
    """
    if not settings.admin_token or token != settings.admin_token:
        logger.warning("Unauthorized admin access attempt")
        raise HTTPException(status_code=403, detail="Forbidden")

    if limit < 0:
        raise HTTPException(status_code=400, detail="limit должно быть >= 0")

    result = rate_limiter.set_limit(ip, limit)
    return result


@router.delete("/delete_ip")
def admin_delete_ip(token: str, ip: str):
    """Удаляет все данные по конкретному IP."""
    logger.info("Delete IP request: token=%s, ip=%s", token[:5] + "..." if len(token) > 5 else token, ip)
    if not settings.admin_token or token != settings.admin_token:
        logger.warning("Unauthorized admin access attempt")
        raise HTTPException(status_code=403, detail="Forbidden")

    rate_limiter.delete_ip(ip)
    logger.info("Successfully deleted IP: %s", ip)
    return {"status": "ok"}



@router.post("/set_default_limit")
def admin_set_default_limit(token: str, limit: int):
    """
    Устанавливает глобальный дефолтный лимит анализов для новых IP.
    /admin/set_default_limit?token=SECRET&limit=3
    """
    if not settings.admin_token or token != settings.admin_token:
        logger.warning("Unauthorized admin access attempt")
        raise HTTPException(status_code=403, detail="Forbidden")

    if limit < 0:
        raise HTTPException(status_code=400, detail="limit должно быть >= 0")

    new_limit = rate_limiter.set_default_limit(limit)
    return {"default_analyze_limit": new_limit}


@router.post("/set_feedback_bonus")
def admin_set_feedback_bonus(token: str, bonus: int):
    """
    Устанавливает глобальный размер бонуса за отзыв.
    /admin/set_feedback_bonus?token=SECRET&bonus=3
    """
    if not settings.admin_token or token != settings.admin_token:
        logger.warning("Unauthorized admin access attempt")
        raise HTTPException(status_code=403, detail="Forbidden")

    if bonus < 0:
        raise HTTPException(status_code=400, detail="bonus должно быть >= 0")

    new_bonus = rate_limiter.set_feedback_bonus(bonus)
    return {"feedback_bonus_analyses": new_bonus}


@router.delete("/clear_all")
def admin_clear_all(token: str):
    """Полностью очищает статистику по всем IP."""
    if not settings.admin_token or token != settings.admin_token:
        logger.warning("Unauthorized admin access attempt")
        raise HTTPException(status_code=403, detail="Forbidden")

    rate_limiter.clear_all()
    return {"status": "ok"}
