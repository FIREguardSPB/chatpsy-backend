"""Storage for analysis results before payment."""
import json
import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from ..config import settings
from ..models.schemas import AnalyzeResponse

logger = logging.getLogger(__name__)

STORAGE_DIR = settings.log_dir / "analysis_storage"
STORAGE_DIR.mkdir(exist_ok=True)


class AnalysisStorage:
    """Stores full analysis results with unique IDs before payment."""

    def __init__(self):
        self._lock = Lock()

    def _generate_analysis_id(self) -> str:
        """Generate unique analysis ID (32 chars hex)."""
        return secrets.token_hex(16)

    def _get_analysis_path(self, analysis_id: str) -> Path:
        """Get file path for analysis."""
        return STORAGE_DIR / f"{analysis_id}.json"

    def save_analysis(
        self,
        analysis: AnalyzeResponse,
        token_usage: Optional[Dict[str, int]],
        client_ip: str,
    ) -> str:
        """
        Save full analysis to storage and return unique ID.
        
        Returns:
            analysis_id: Unique identifier to retrieve analysis later
        """
        analysis_id = self._generate_analysis_id()
        
        with self._lock:
            data = {
                "analysis_id": analysis_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "client_ip": client_ip,
                "paid": False,
                "paid_at": None,
                "token_usage": token_usage,
                "analysis": analysis.model_dump(mode="json"),
            }

            analysis_path = self._get_analysis_path(analysis_id)
            with open(analysis_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(
            "Analysis saved: id=%s, ip=%s, tokens=%s",
            analysis_id,
            client_ip,
            token_usage,
        )
        return analysis_id

    def get_analysis(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis data by ID."""
        analysis_path = self._get_analysis_path(analysis_id)
        
        if not analysis_path.exists():
            return None

        try:
            with open(analysis_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.exception("Failed to load analysis id=%s: %r", analysis_id, exc)
            return None

    def mark_as_paid(self, analysis_id: str) -> bool:
        """Mark analysis as paid."""
        with self._lock:
            data = self.get_analysis(analysis_id)
            if not data:
                return False

            data["paid"] = True
            data["paid_at"] = datetime.now(timezone.utc).isoformat()

            analysis_path = self._get_analysis_path(analysis_id)
            with open(analysis_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("Analysis marked as paid: id=%s", analysis_id)
        return True

    def is_paid(self, analysis_id: str) -> bool:
        """Check if analysis has been paid for."""
        data = self.get_analysis(analysis_id)
        return data["paid"] if data else False

    def cleanup_old_analyses(self, days: int = 7) -> int:
        """
        Remove unpaid analyses older than N days.
        Returns number of deleted files.
        """
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        deleted = 0

        with self._lock:
            for analysis_file in STORAGE_DIR.glob("*.json"):
                try:
                    with open(analysis_file, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    created_at = datetime.fromisoformat(data["created_at"])
                    if not data["paid"] and created_at < cutoff:
                        analysis_file.unlink()
                        deleted += 1
                except Exception as exc:
                    logger.exception("Error cleaning up %s: %r", analysis_file, exc)

        logger.info("Cleaned up %d old unpaid analyses", deleted)
        return deleted


# Global storage instance
analysis_storage = AnalysisStorage()
