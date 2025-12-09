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