"""In-memory LRU-style cache backend — used when Redis is unavailable."""

from __future__ import annotations

import json
import logging
import time
import threading
from typing import Any

logger = logging.getLogger(__name__)

_SENTINEL = object()


class _Entry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: str, ttl: int) -> None:
        self.value = value
        self.expires_at = time.monotonic() + ttl if ttl > 0 else float("inf")

    @property
    def expired(self) -> bool:
        return time.monotonic() > self.expires_at


class InMemoryCache:
    """
    Thread-safe in-process dict cache that mirrors the RedisCache API.
    Used as an automatic fallback when Redis is not configured.

    Eviction: expired entries are purged lazily on every set/get.
    Max size: capped at `max_size` entries (oldest inserted entry dropped first).
    """

    def __init__(self, max_size: int = 512) -> None:
        self._store: dict[str, _Entry] = {}
        self._lock = threading.Lock()
        self._max_size = max_size
        self._available: bool = True  # always available

    # ── Public API (matches RedisCache) ──────────────────────
    def get_json(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key, _SENTINEL)
            if entry is _SENTINEL:
                return None
            if entry.expired:
                del self._store[key]
                return None
            try:
                return json.loads(entry.value)
            except Exception:  # noqa: BLE001
                return None

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> bool:
        try:
            serialised = json.dumps(value, ensure_ascii=False, default=str)
        except Exception as exc:  # noqa: BLE001
            logger.warning("InMemoryCache: serialisation failed — %s", exc)
            return False

        with self._lock:
            self._evict_expired()
            if len(self._store) >= self._max_size and key not in self._store:
                # Drop oldest entry
                oldest_key = next(iter(self._store))
                del self._store[oldest_key]
            self._store[key] = _Entry(serialised, ttl_seconds)
            return True

    # ── Internal ──────────────────────────────────────────────
    def _evict_expired(self) -> None:
        """Remove all expired entries (called while holding _lock)."""
        expired_keys = [k for k, v in self._store.items() if v.expired]
        for k in expired_keys:
            del self._store[k]
