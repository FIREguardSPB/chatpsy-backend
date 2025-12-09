"""Statistics route."""
from fastapi import APIRouter, Depends

from ...config import settings
from ...core.rate_limiter import rate_limiter
from ...dependencies import get_client_ip

router = APIRouter()


@router.get("/usage_stats")
def usage_stats():
    """
    Простая статистика:
    - общий объём загруженных данных
    - список IP: сколько запросов, сколько МБ, лимиты анализов
    """
    return rate_limiter.get_all_stats()


@router.get("/my_stats")
def my_stats(client_ip: str = Depends(get_client_ip)):
    return rate_limiter.get_ip_stats(client_ip)
@router.get("/debug/config")
def debug_config():
    """Debug endpoint to check loaded configuration."""
    return {
        "use_llm": settings.use_llm,
        "openai_model": settings.openai_model,
        "openai_base_url": settings.openai_base_url,
        "payment_enabled": settings.payment_enabled,
        "payment_provider": settings.payment_provider,
        "payment_test_mode": settings.payment_test_mode,
        "feedback_bonus_analyses": getattr(rate_limiter, "_feedback_bonus", settings.feedback_bonus_analyses),
        "llm_max_chars": settings.llm_max_chars,
        "llm_max_tokens": settings.llm_max_tokens,
    }
