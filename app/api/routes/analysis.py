"""Analysis routes: /chat_meta and /analyze_chat."""
import logging
from fastapi import APIRouter, Depends, HTTPException, Request

from ...config import settings
from ...core.rate_limiter import rate_limiter
from ...core.analysis_storage import analysis_storage
from ...dependencies import get_client_ip
from ...models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatMetaRequest,
    ChatMetaResponse,
    ParticipantProfile,
    RelationshipSummary,
    Recommendation,
    ChatStats,
)
from ...services.chat_analyzer import analyze_chat_text
from ...services.chat_parser import parse_chat_text
from ...services.preview_service import create_preview

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat_meta", response_model=ChatMetaResponse)
async def chat_meta(
    body: ChatMetaRequest,
    request: Request,
    client_ip: str = Depends(get_client_ip),
):
    """
    Быстрая статистика без LLM.
    Возвращает мета-информацию о чате.
    """
    text = body.chat_text

    if not text.strip():
        raise HTTPException(status_code=400, detail="Пустой текст переписки")

    upload_bytes = len(text.encode("utf-8"))

    # Просто считаем статистику без LLM
    try:
        _, stats = parse_chat_text(text)
    except Exception as exc:
        logger.exception("Ошибка при расчёте мета-информации")
        raise HTTPException(
            status_code=500, detail="Ошибка при расчёте мета-информации"
        ) from exc

    # Прикидываем, сколько реально уйдёт в модель (обрезка сниппета)
    snippet_chars = min(len(text), settings.llm_max_chars)
    snippet_bytes = len(text[:snippet_chars].encode("utf-8"))

    # Трекаем запрос
    rate_limiter.track_request(client_ip, upload_bytes)

    logger.info(
        "chat_meta ip=%s upload_bytes=%d snippet_bytes=%d",
        client_ip,
        upload_bytes,
        snippet_bytes,
    )

    return ChatMetaResponse(
        stats=stats,
        upload_bytes=upload_bytes,
        snippet_bytes=snippet_bytes,
        recommended_bytes=settings.recommended_bytes,
    )


@router.post("/analyze_chat", response_model=AnalyzeResponse)
async def analyze_chat(
    body: AnalyzeRequest,
    request: Request,
    client_ip: str = Depends(get_client_ip),
):
    """
    Полный анализ переписки с LLM.
    """
    text = body.chat_text

    if not text.strip():
        raise HTTPException(status_code=400, detail="Пустой текст переписки")

    # Простая проверка на бинарные данные: нулевые байты
    if "\x00" in text:
        raise HTTPException(
            status_code=400,
            detail=(
                "Похоже, вы загрузили бинарный файл. "
                "Поддерживаются только .txt и .html экспорты чатов."
            ),
        )

    size_bytes = len(text.encode("utf-8"))

    # Проверяем лимит и учитываем использование
    if not settings.payment_enabled:
        if not rate_limiter.check_and_increment_analysis(client_ip):
            ip_stats = rate_limiter.get_ip_stats(client_ip)
            logger.info(
                "Rate limit exceeded for ip=%s used=%d limit=%d",
                client_ip,
                ip_stats["analyze_used"],
                ip_stats["analyze_limit"],
            )
            raise HTTPException(
                status_code=429,
                detail=(
                    "Тестовый лимит анализов исчерпан для этого устройства. "
                    "Напишите нам, и мы добавим ещё запросы."
                ),
            )
    else:
        # Режим оплаты: проверяем остаток (не использованные анализы) ДО инкремента
        ip_stats_before = rate_limiter.get_ip_stats(client_ip)
        used_before = ip_stats_before["analyze_used"]
        limit = ip_stats_before["analyze_limit"]
        remaining_before = limit - used_before  # сколько осталось бесплатных
        
        # Инкрементируем счётчик
        rate_limiter.increment_analysis_used(client_ip)

    # Трекаем запрос
    rate_limiter.track_request(client_ip, size_bytes)

    try:
        response, token_usage = analyze_chat_text(
            chat_text=text,
            from_date=body.range_from,
            to_date=body.range_to,
        )

        # Если анализ упал в заглушку - откатываем инкремент (не учитываем запрос)
        if response.is_fallback:
            logger.warning(
                "Analysis failed with fallback for ip=%s, reverting usage increment",
                client_ip,
            )
            # Откатываем analyze_used на 1
            if not settings.payment_enabled:
                rate_limiter.decrement_analysis_used(client_ip)
            else:
                rate_limiter.decrement_analysis_used(client_ip)

        ip_stats = rate_limiter.get_ip_stats(client_ip)
        logger.info(
            "analyze_chat ok ip=%s bytes=%d used=%d limit=%d tokens=%s is_fallback=%s",
            client_ip,
            size_bytes,
            ip_stats["analyze_used"],
            ip_stats["analyze_limit"],
            token_usage,
            response.is_fallback,
        )

        # Сохраняем полный анализ и возвращаем preview (если включены платежи)
        if settings.payment_enabled:
            # Проверяем, был ли остаток ДО текущего запроса
            if remaining_before > 0:
                # Был остаток бесплатных анализов: возвращаем полный результат
                logger.info(
                    "Freebies available for ip=%s, remaining=%d, returning full analysis",
                    client_ip,
                    remaining_before,
                )
                return response
            else:
                # Остатка не было: сохраняем и возвращаем preview
                analysis_id = analysis_storage.save_analysis(
                    response, token_usage, client_ip
                )
                preview = create_preview(response)
                preview.analysis_id = analysis_id
                logger.info(
                    "No freebies for ip=%s, remaining=%d, preview generated for analysis_id=%s",
                    client_ip,
                    remaining_before,
                    analysis_id,
                )
                return preview
        else:
            # Платежи отключены - возвращаем полный анализ
            return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Ошибка при анализе переписки")
        raise HTTPException(
            status_code=500, detail="Ошибка при анализе переписки"
        ) from exc
