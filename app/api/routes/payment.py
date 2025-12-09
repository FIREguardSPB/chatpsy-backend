"""Payment routes."""
import hashlib
import hmac
import logging

from fastapi import APIRouter, HTTPException, Request
# YooKassa will be imported lazily in handlers

from ...config import settings
from ...core.analysis_storage import analysis_storage
from ...models.schemas import (
    AnalyzeResponse,
    FullAnalysisRequest,
    PaymentCreateRequest,
    PaymentCreateResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/create_payment", response_model=PaymentCreateResponse)
async def create_payment(body: PaymentCreateRequest):
    """
    Create payment for full analysis access.
    Returns payment URL to redirect user.
    """
    analysis_id = body.analysis_id

    # Проверяем что анализ существует
    data = analysis_storage.get_analysis(analysis_id)
    if not data:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Проверяем не оплачено ли уже
    if data["paid"]:
        raise HTTPException(status_code=400, detail="Already paid")

    # Интеграция платежа
    if (
        settings.payment_provider == "yookassa"
        and settings.yookassa_shop_id
        and settings.yookassa_secret_key
    ):
        # Configure YooKassa
        from yookassa import Configuration, Payment
        Configuration.account_id = settings.yookassa_shop_id
        Configuration.secret_key = settings.yookassa_secret_key

        payment = Payment.create({
            "amount": {"value": str(settings.payment_price_rub), "currency": "RUB"},
            "confirmation": {"type": "redirect", "return_url": body.return_url},
            "capture": True,
            "description": f"Анализ переписки {analysis_id}",
            "metadata": {"analysis_id": analysis_id},
        })
        payment_id = payment.id
        payment_url = payment.confirmation.confirmation_url
    else:
        # ЗАГЛУШКА для разработки
        payment_id = f"test_payment_{analysis_id[:8]}"
        payment_url = f"https://payment.example.com/pay?id={payment_id}&return={body.return_url}"

    logger.info(
        "Payment created: analysis_id=%s payment_id=%s",
        analysis_id,
        payment_id,
    )

    return PaymentCreateResponse(
        payment_url=payment_url,
        payment_id=payment_id,
    )


@router.post("/payment_webhook")
async def payment_webhook(request: Request):
    """
    Webhook от платёжной системы.
    Вызывается после успешной оплаты.
    """
    body = await request.json()

    logger.info("Payment webhook received: %s", body)

    # TODO: Верификация подписи webhook (зависит от провайдера)
    # Пример для ЮKassa:
    # signature = request.headers.get("X-Signature")
    # if not verify_yookassa_signature(body, signature):
    #     raise HTTPException(status_code=403, detail="Invalid signature")

    # Извлекаем данные (структура зависит от провайдера)
    # Пример для ЮKassa:
    # payment_id = body.get("object", {}).get("id")
    # status = body.get("object", {}).get("status")
    # analysis_id = body.get("object", {}).get("metadata", {}).get("analysis_id")

    # YooKassa webhook parsing
    if settings.payment_provider == "yookassa" and isinstance(body, dict) and body.get("event"):
        obj = body.get("object") or {}
        payment_id = obj.get("id")
        status = obj.get("status")
        metadata = obj.get("metadata") or {}
        analysis_id = metadata.get("analysis_id")
    else:
        # ЗАГЛУШКА для разработки
        payment_id = body.get("payment_id")
        status = body.get("status")
        analysis_id = body.get("analysis_id")

    if not all([payment_id, status, analysis_id]):
        raise HTTPException(status_code=400, detail="Missing required fields")

    # Если оплата успешна - открываем доступ
    if status in ["succeeded", "success"]:
        success = analysis_storage.mark_as_paid(analysis_id)
        if not success:
            logger.error("Failed to mark analysis as paid: %s", analysis_id)
            raise HTTPException(status_code=404, detail="Analysis not found")

        logger.info(
            "Payment successful: analysis_id=%s payment_id=%s",
            analysis_id,
            payment_id,
        )

        return {"status": "ok", "analysis_id": analysis_id}

    return {"status": "pending"}


@router.post("/get_full_analysis", response_model=AnalyzeResponse)
async def get_full_analysis(body: FullAnalysisRequest):
    """
    Get full analysis after payment.
    Requires analysis to be marked as paid.
    """
    analysis_id = body.analysis_id

    # Загружаем анализ
    data = analysis_storage.get_analysis(analysis_id)
    if not data:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Проверяем оплачен ли
    if not data["paid"]:
        raise HTTPException(
            status_code=402,  # Payment Required
            detail="Payment required to access full analysis",
        )

    # Возвращаем полный анализ
    analysis_dict = data["analysis"]
    full_analysis = AnalyzeResponse(**analysis_dict)
    full_analysis.is_preview = False
    full_analysis.payment_required = False
    full_analysis.analysis_id = analysis_id

    logger.info("Full analysis retrieved: analysis_id=%s", analysis_id)

    return full_analysis


def verify_yookassa_signature(payload: dict, signature: str) -> bool:
    """
    Verify YooKassa webhook signature.
    https://yookassa.ru/developers/using-api/webhooks#webhook-authenticity
    """
    if not settings.payment_webhook_secret or not signature:
        return False

    # Строка для подписи (зависит от провайдера)
    message = f"{payload}"  # Simplified, нужна правильная сериализация

    expected_signature = hmac.new(
        settings.payment_webhook_secret.encode(),
        message.encode(),
        hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(signature, expected_signature)
