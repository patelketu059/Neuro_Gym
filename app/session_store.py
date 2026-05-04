"""Redis-backed session store with transparent in-memory fallback.

If Redis is unavailable at startup the store falls back to an in-memory dict
with TTL-based eviction — the application boots even when Redis isn't running.

Environment variable:
    REDIS_URL — connection string (default: redis://localhost:6379)

Redis setup (local dev):
    docker run -d --name redis -p 6379:6379 redis:7-alpine
"""
from __future__ import annotations
import json
import logging
import os
import time
from typing import Optional

from config.rag_config import SESSION_TTL_SECONDS, REDIS_URL_DEFAULT

log = logging.getLogger(__name__)
_TTL = int(SESSION_TTL_SECONDS)


class SessionStore:
    """Unified session persistence — Redis when available, in-memory fallback."""

    def __init__(self, redis_url: str = REDIS_URL_DEFAULT) -> None:
        self._redis = None
        self._mem: dict[str, dict] = {}  # key → {"data": dict, "ts": float}
        try:
            import redis as _redis
            r = _redis.from_url(
                redis_url,
                decode_responses=True,
                socket_connect_timeout=1,
            )
            r.ping()
            self._redis = r
            log.info("[SessionStore] Redis backend connected at %s", redis_url)
        except Exception as exc:
            log.warning(
                "[SessionStore] Redis unavailable (%s) — using in-memory fallback", exc
            )

    @property
    def backend(self) -> str:
        return "redis" if self._redis else "memory"

    # ── CRUD ─────────────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[dict]:
        if self._redis:
            raw = self._redis.get(f"session:{key}")
            return json.loads(raw) if raw else None
        entry = self._mem.get(key)
        if not entry:
            return None
        if time.time() - entry["ts"] > SESSION_TTL_SECONDS:
            del self._mem[key]
            return None
        return entry["data"]

    def set(self, key: str, data: dict) -> None:
        if self._redis:
            self._redis.setex(f"session:{key}", _TTL, json.dumps(data))
        else:
            self._mem[key] = {"data": data, "ts": time.time()}

    def delete(self, key: str) -> None:
        if self._redis:
            self._redis.delete(f"session:{key}")
        else:
            self._mem.pop(key, None)

    def touch(self, key: str) -> None:
        """Reset TTL without rewriting session data."""
        if self._redis:
            self._redis.expire(f"session:{key}", _TTL)
        elif key in self._mem:
            self._mem[key]["ts"] = time.time()

    def evict_stale(self) -> None:
        """Purge expired in-memory sessions. No-op for Redis (TTL is automatic)."""
        if self._redis:
            return
        cutoff = time.time() - SESSION_TTL_SECONDS
        stale = [k for k, v in self._mem.items() if v["ts"] < cutoff]
        for k in stale:
            del self._mem[k]


# ── Module-level singleton ────────────────────────────────────────────────────

_store: Optional[SessionStore] = None


def get_store() -> SessionStore:
    global _store
    if _store is None:
        url = os.environ.get("REDIS_URL", REDIS_URL_DEFAULT)
        _store = SessionStore(url)
    return _store
