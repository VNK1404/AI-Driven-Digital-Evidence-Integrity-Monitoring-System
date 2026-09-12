"""Small fail-open Redis adapter used by the cache manager."""

from __future__ import annotations

import json
import logging
import socket
from typing import Any
from urllib.parse import urlparse

from fake_news_module.config import CACHE_ENABLED, REDIS_URL

logger = logging.getLogger(__name__)


class RedisCache:
    """JSON Redis cache wrapper that never raises into business logic."""

    def __init__(self, redis_url: str = REDIS_URL, client: Any | None = None) -> None:
        self._redis_url = redis_url
        self._client = client
        self._available: bool | None = None

    @property
    def client(self) -> Any | None:
        if self._client is not None:
            return self._client
        if not CACHE_ENABLED or not self._redis_url:
            return None
        if self._available is False:
            return None
        try:
            import redis

            if not self._tcp_available():
                self._available = False
                return None

            self._client = redis.Redis.from_url(
                self._redis_url,
                socket_connect_timeout=0.05,
                socket_timeout=0.05,
                decode_responses=True,
                retry_on_timeout=False,
            )
            self._client.ping()
            self._available = True
            return self._client
        except Exception as exc:  # noqa: BLE001
            self._available = False
            self._client = None
            logger.warning("Redis unavailable; continuing without cache: %s", exc)
            return None

    def _tcp_available(self) -> bool:
        parsed = urlparse(self._redis_url)
        if parsed.scheme not in {"redis", "rediss"}:
            return True
        host = parsed.hostname or "localhost"
        if host.lower() == "localhost":
            host = "127.0.0.1"
        port = parsed.port or 6379
        try:
            with socket.create_connection((host, port), timeout=0.001):
                return True
        except OSError as exc:
            logger.warning("Redis unavailable; continuing without cache: %s", exc)
            return False

    def get_json(self, key: str) -> Any | None:
        client = self.client
        if client is None:
            return None
        try:
            payload = client.get(key)
            if payload is None:
                return None
            return json.loads(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cache failure reading key '%s': %s", key, exc)
            return None

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> bool:
        client = self.client
        if client is None:
            return False
        try:
            client.setex(key, ttl_seconds, json.dumps(value, ensure_ascii=False))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cache failure writing key '%s': %s", key, exc)
            return False
