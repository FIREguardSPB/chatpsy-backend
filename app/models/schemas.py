from datetime import datetime, date
from typing import Dict, List, Optional

from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    chat_text: str
    # опциональные границы периода анализа (по дате, без времени)
    range_from: Optional[date] = None
    range_to: Optional[date] = None


class ParticipantProfile(BaseModel):
    id: str
    display_name: str
    traits: Dict[str, str]
    summary: str


class RelationshipSummary(BaseModel):
    description: str
    red_flags: List[str]
    green_flags: List[str]


class Recommendation(BaseModel):
    title: str
    text: str


class ParticipantStats(BaseModel):
    id: str
    messages_count: int
    avg_message_length: float


class ChatStats(BaseModel):
    total_messages: int
    participants: List[ParticipantStats]
    first_message_at: Optional[datetime] = None
    last_message_at: Optional[datetime] = None


class AnalyzeResponse(BaseModel):
    participants: List[ParticipantProfile]
    relationship: RelationshipSummary
    recommendations: List[Recommendation]
    stats: ChatStats
    # Payment fields
    analysis_id: Optional[str] = None  # Unique ID for retrieving full analysis
    is_preview: bool = False  # Whether this is a preview (partial) result
    payment_required: bool = False  # Whether payment is needed for full access
    # Error handling fields
    is_fallback: bool = False  # Whether this is a fallback/stub response due to error
    error_message: Optional[str] = None  # Technical error details for debugging

class ChatMetaRequest(BaseModel):
    chat_text: str


class ChatMetaResponse(BaseModel):
    stats: ChatStats
    snippet_bytes: int
    upload_bytes: int          # фактический размер текста
    recommended_bytes: int     # рекомендуемый лимит (например, ~1 МБ)

class FeedbackRequest(BaseModel):
    text: str
    contact: Optional[str] = None


class PaymentCreateRequest(BaseModel):
    analysis_id: str
    return_url: str  # URL to redirect after payment


class PaymentCreateResponse(BaseModel):
    payment_url: str  # URL to redirect user for payment
    payment_id: str  # Unique payment ID


class PaymentWebhookRequest(BaseModel):
    """Webhook from payment provider (structure depends on provider)."""
    payment_id: str
    status: str  # e.g., "success", "failed"
    analysis_id: str
    signature: Optional[str] = None  # For webhook verification


class FullAnalysisRequest(BaseModel):
    analysis_id: str