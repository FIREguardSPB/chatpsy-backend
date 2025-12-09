"""Service for creating preview (partial) analysis results."""
import logging
from typing import List

from ..models.schemas import (
    AnalyzeResponse,
    ParticipantProfile,
    Recommendation,
    RelationshipSummary,
)

logger = logging.getLogger(__name__)

PREVIEW_PERCENTAGE = 0.1  # 10% preview


def _truncate_text(text: str, ratio: float = PREVIEW_PERCENTAGE) -> str:
    """Truncate text to specified ratio and add ellipsis."""
    if not text:
        return text
    
    max_len = max(int(len(text) * ratio), 50)  # Minimum 50 chars
    if len(text) <= max_len:
        return text
    
    return text[:max_len] + "..."


def _truncate_list(items: List[str], ratio: float = PREVIEW_PERCENTAGE) -> List[str]:
    """Keep only first N% of list items."""
    if not items:
        return items
    
    keep_count = max(int(len(items) * ratio), 1)  # Minimum 1 item
    return items[:keep_count]


def create_preview(full_analysis: AnalyzeResponse) -> AnalyzeResponse:
    """
    Create preview version of analysis (10% of content).
    
    Truncates:
    - Participant summaries to 10%
    - Relationship description to 10%
    - Red/green flags to first item only
    - Recommendations to first one only
    """
    
    # Truncate participant profiles
    preview_participants: List[ParticipantProfile] = []
    for p in full_analysis.participants:
        preview_participants.append(
            ParticipantProfile(
                id=p.id,
                display_name=p.display_name,
                traits=p.traits,  # Keep traits intact
                summary=_truncate_text(p.summary),  # Truncate summary
            )
        )
    
    # Truncate relationship
    preview_relationship = RelationshipSummary(
        description=_truncate_text(full_analysis.relationship.description),
        red_flags=_truncate_list(full_analysis.relationship.red_flags),
        green_flags=_truncate_list(full_analysis.relationship.green_flags),
    )
    
    # Keep only first recommendation
    preview_recommendations: List[Recommendation] = []
    if full_analysis.recommendations:
        first_rec = full_analysis.recommendations[0]
        preview_recommendations.append(
            Recommendation(
                title=first_rec.title,
                text=_truncate_text(first_rec.text),
            )
        )
    
    return AnalyzeResponse(
        participants=preview_participants,
        relationship=preview_relationship,
        recommendations=preview_recommendations,
        stats=full_analysis.stats,  # Stats stay intact
        is_preview=True,
        payment_required=True,
    )
