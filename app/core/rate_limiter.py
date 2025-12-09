"""IP-based rate limiting logic."""
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict

from ..config import settings

logger = logging.getLogger(__name__)

IP_STATS_PATH = settings.log_dir / "ip_stats.json"
CONFIG_PATH = settings.log_dir / "rate_limiter_config.json"


def _default_ip_record() -> Dict[str, Any]:
    """Базовая структура записи по IP."""
    return {
        "requests": 0,
        "bytes": 0,
        "first_seen": None,
        "last_seen": None,
        "analyze_used": 0,
        "analyze_limit": settings.default_analyze_limit,
        "feedback_bonus_used": False,
    }


class RateLimiter:
    """Manages IP-based rate limiting with persistent storage."""

    def __init__(self):
        self._lock = Lock()
        self._default_limit = settings.default_analyze_limit
        self._feedback_bonus = settings.feedback_bonus_analyses
        self._ip_stats: Dict[str, Dict[str, Any]] = defaultdict(self._make_default_record)
        self._load_config()
        self._load_from_file()

    def _make_default_record(self) -> Dict[str, Any]:
        """Базовая структура записи по IP с учётом текущего глобального лимита."""
        return {
            "requests": 0,
            "bytes": 0,
            "first_seen": None,
            "last_seen": None,
            "analyze_used": 0,
            "analyze_limit": self._default_limit,
            "feedback_bonus_used": False,
        }

    def _load_from_file(self) -> None:
        """Загружаем ip_stats.json, если он есть."""
        if not IP_STATS_PATH.exists():
            return

        try:
            with open(IP_STATS_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as exc:
            logger.exception("Не удалось прочитать ip_stats.json: %r", exc)
            return

        for ip, v in raw.items():
            rec = _default_ip_record()
            rec["requests"] = int(v.get("requests", 0))
            rec["bytes"] = int(v.get("bytes", 0))
            rec["analyze_used"] = int(v.get("analyze_used", 0))
            rec["analyze_limit"] = int(
                v.get("analyze_limit", settings.default_analyze_limit)
            )
            rec["feedback_bonus_used"] = bool(v.get("feedback_bonus_used", False))

            fs = v.get("first_seen")
            ls = v.get("last_seen")
            rec["first_seen"] = datetime.fromisoformat(fs) if fs else None
            rec["last_seen"] = datetime.fromisoformat(ls) if ls else None

            self._ip_stats[ip] = rec

        logger.info("ip_stats.json loaded, records: %d", len(self._ip_stats))

    def _save_to_file(self) -> None:
        """Сохраняем _ip_stats в ip_stats.json."""
        try:
            with self._lock:
                snapshot: Dict[str, Dict[str, Any]] = {}
                for ip, v in self._ip_stats.items():
                    snapshot[ip] = {
                        "requests": v["requests"],
                        "bytes": v["bytes"],
                        "analyze_used": v["analyze_used"],
                        "analyze_limit": v["analyze_limit"],
                        "feedback_bonus_used": v.get("feedback_bonus_used", False),
                        "first_seen": (
                            v["first_seen"].isoformat() if v["first_seen"] else None
                        ),
                        "last_seen": (
                            v["last_seen"].isoformat() if v["last_seen"] else None
                        ),
                    }

            with open(IP_STATS_PATH, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.exception("Не удалось сохранить ip_stats.json: %r", exc)

    def track_request(self, client_ip: str, size_bytes: int = 0) -> None:
        """Регистрирует запрос от IP."""
        now = datetime.now(timezone.utc)
        with self._lock:
            rec = self._ip_stats[client_ip]
            rec["requests"] += 1
            rec["bytes"] += size_bytes
            if rec["first_seen"] is None:
                rec["first_seen"] = now
            rec["last_seen"] = now
        self._save_to_file()

    def check_and_increment_analysis(self, client_ip: str) -> bool:
        """
        Проверяет лимит анализов для IP и инкрементирует счетчик если можно.
        Возвращает True если лимит не превышен, False если превышен.
        """
        with self._lock:
            rec = self._ip_stats[client_ip]
            used = rec["analyze_used"]
            limit = rec["analyze_limit"]

            if used >= limit:
                return False

            rec["analyze_used"] += 1

        self._save_to_file()
        return True

    def increment_analysis_used(self, client_ip: str) -> None:
        """Безусловно увеличивает счетчик использованных анализов для IP."""
        with self._lock:
            rec = self._ip_stats[client_ip]
            rec["analyze_used"] += 1
        self._save_to_file()

    def decrement_analysis_used(self, client_ip: str) -> None:
        """Откатывает счетчик использованных анализов (например, при ошибке)."""
        with self._lock:
            rec = self._ip_stats[client_ip]
            if rec["analyze_used"] > 0:
                rec["analyze_used"] -= 1
        self._save_to_file()

    def get_ip_stats(self, client_ip: str) -> Dict[str, Any]:
        """Возвращает статистику по IP."""
        with self._lock:
            return dict(self._ip_stats[client_ip])

    def add_credits(self, client_ip: str, n: int) -> Dict[str, Any]:
        """Добавляет n анализов для IP."""
        with self._lock:
            rec = self._ip_stats[client_ip]
            rec["analyze_limit"] += n
            logger.info(
                "Admin add credits ip=%s n=%d new_limit=%d",
                client_ip,
                n,
                rec["analyze_limit"],
            )
            result = {
                "ip": client_ip,
                "analyze_limit": rec["analyze_limit"],
                "analyze_used": rec["analyze_used"],
            }

        self._save_to_file()
        return result

    def set_limit(self, client_ip: str, limit: int) -> Dict[str, Any]:
        """Устанавливает абсолютный лимит анализов для IP."""
        if limit < 0:
            limit = 0
        with self._lock:
            rec = self._ip_stats[client_ip]
            rec["analyze_limit"] = limit
            logger.info(
                "Admin set limit ip=%s limit=%d",
                client_ip,
                limit,
            )
            result = {
                "ip": client_ip,
                "analyze_limit": rec["analyze_limit"],
                "analyze_used": rec["analyze_used"],
            }
        self._save_to_file()
        return result

    def grant_feedback_bonus(self, client_ip: str) -> int:
        """
        Начисляет бонус за фидбэк если ещё не было.
        Возвращает количество начисленных анализов.
        
        Логика: бонус гарантирует, что у пользователя будет минимум feedback_bonus бесплатных анализов.
        Если у него уже есть больше, не уменьшаем. Если меньше - добавляем до feedback_bonus.
        """
        with self._lock:
            rec = self._ip_stats[client_ip]
            already_used = rec.get("feedback_bonus_used", False)

            if already_used or self._feedback_bonus <= 0:
                granted = 0
            else:
                # Текущий баланс
                used = rec["analyze_used"]
                current_limit = rec["analyze_limit"]
                balance = current_limit - used  # сколько осталось бесплатных
                
                # Гарантируем, что баланс будет минимум feedback_bonus
                if balance < self._feedback_bonus:
                    need_to_add = self._feedback_bonus - balance
                    rec["analyze_limit"] += need_to_add
                    granted = need_to_add
                else:
                    # У пользователя уже есть достаточно бесплатных
                    granted = 0
                
                rec["feedback_bonus_used"] = True
                logger.info(
                    "Feedback bonus granted for ip=%s: balance_before=%d, bonus=%d, granted=%d, new_limit=%d",
                    client_ip,
                    balance,
                    self._feedback_bonus,
                    granted,
                    rec["analyze_limit"],
                )

        self._save_to_file()
        return granted

    def get_all_stats(self) -> Dict[str, Any]:
        """Возвращает общую статистику по всем IP."""
        with self._lock:
            total_bytes = sum(v["bytes"] for v in self._ip_stats.values())
            clients = []
            for ip, v in self._ip_stats.items():
                mb = v["bytes"] / (1024 * 1024)
                first_seen = v["first_seen"]
                last_seen = v["last_seen"]
                clients.append(
                    {
                        "ip": ip,
                        "requests": v["requests"],
                        "bytes": v["bytes"],
                        "megabytes": round(mb, 3),
                        "first_seen": first_seen.isoformat() if first_seen else None,
                        "last_seen": last_seen.isoformat() if last_seen else None,
                        "analyze_used": v["analyze_used"],
                        "analyze_limit": v["analyze_limit"],
                        "feedback_bonus_used": v.get("feedback_bonus_used", False),
                    }
                )

        return {
            "total_bytes": total_bytes,
            "total_megabytes": round(total_bytes / (1024 * 1024), 3),
            "clients": clients,
        }

    def _load_config(self) -> None:
        """Загружает глобальный лимит из rate_limiter_config.json, если он есть."""
        if not CONFIG_PATH.exists():
            return
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self._default_limit = int(cfg.get("default_analyze_limit", self._default_limit))
            self._feedback_bonus = int(cfg.get("feedback_bonus_analyses", self._feedback_bonus))
            logger.info("Loaded limits from config: default=%d, feedback_bonus=%d", self._default_limit, self._feedback_bonus)
        except Exception as exc:
            logger.exception("Не удалось прочитать rate_limiter_config.json: %r", exc)

    def _save_config(self) -> None:
        """Сохраняет глобальный лимит в rate_limiter_config.json."""
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump({
                    "default_analyze_limit": self._default_limit,
                    "feedback_bonus_analyses": self._feedback_bonus,
                }, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.exception("Не удалось сохранить rate_limiter_config.json: %r", exc)

    def set_feedback_bonus(self, bonus: int) -> int:
        """Устанавливает глобальный размер бонуса за отзыв."""
        if bonus < 0:
            bonus = 0
        with self._lock:
            self._feedback_bonus = bonus
            logger.info("Admin set feedback bonus = %d", bonus)
            self._save_config()
        return self._feedback_bonus

    def set_default_limit(self, limit: int) -> int:
        """Устанавливает глобальный дефолтный лимит для новых IP."""
        if limit < 0:
            limit = 0
        with self._lock:
            self._default_limit = limit
            logger.info("Admin set global default analyze limit = %d", limit)
            self._save_config()
        return self._default_limit

    def delete_ip(self, client_ip: str) -> None:
        """Полностью удаляет статистику по IP."""
        with self._lock:
            if client_ip in self._ip_stats:
                self._ip_stats.pop(client_ip)
                logger.info("Admin delete ip=%s", client_ip)
        self._save_to_file()

    def clear_all(self) -> None:
        """Очищает все данные по IP."""
        with self._lock:
            self._ip_stats.clear()
            logger.info("Admin clear all ip stats")
        self._save_to_file()


# Глобальный экземпляр
rate_limiter = RateLimiter()
