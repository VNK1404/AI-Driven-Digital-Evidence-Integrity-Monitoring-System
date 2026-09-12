"""Stable cache-key generation and typed cache operations."""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from fake_news_module.cache.redis_cache import RedisCache
from fake_news_module.config import CACHE_KEY_PREFIX, CACHE_TTL_SECONDS

logger = logging.getLogger(__name__)


def _make_backend():
    """
    Return the best available cache backend.

    Priority:
        1. Redis (if REDIS_URL is set and the server is reachable)
        2. InMemoryCache (always available — no external dependency)
    """
    redis = RedisCache()
    # Touch .client to trigger the connectivity check
    if redis.client is not None:
        logger.info("CacheManager: Redis backend active (%s)", redis._redis_url)
        return redis

    # Redis unavailable — fall back to in-process memory cache
    from fake_news_module.cache.memory_cache import InMemoryCache
    logger.info("CacheManager: Redis not available — using in-memory cache fallback")
    return InMemoryCache()


class CacheManager:
    """High-level cache facade for pipeline, API, and similarity results.

    Transparently uses Redis when available, or falls back to an in-process
    InMemoryCache so cache writes never fail silently.
    """

    def __init__(
        self,
        backend=None,
        ttl_seconds: int = CACHE_TTL_SECONDS,
        key_prefix: str = CACHE_KEY_PREFIX,
    ) -> None:
        self._backend = backend or _make_backend()
        self.ttl_seconds = ttl_seconds
        self.key_prefix = key_prefix

    # Keep redis_cache as an alias for any code that accesses it directly
    @property
    def redis_cache(self):
        return self._backend

    @staticmethod
    def stable_hash(value: str) -> str:
        normalized = " ".join((value or "").strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def claim_hash(self, claim: str) -> str:
        return self.stable_hash(claim)

    def api_hash(self, api_name: str, claim: str) -> str:
        return self.stable_hash(f"{api_name}:{claim}")

    def _key(self, namespace: str, digest: str) -> str:
        return f"{self.key_prefix}:{namespace}:{digest}"

    # ── Public typed accessors ────────────────────────────────

    def get_pipeline_result(self, claim: str) -> dict[str, Any] | None:
        return self._get("pipeline", self.claim_hash(claim))

    def set_pipeline_result(self, claim: str, result: dict[str, Any]) -> bool:
        return self._set("pipeline", self.claim_hash(claim), result)

    def get_api_response(self, api_name: str, claim: str) -> Any | None:
        return self._get(f"api:{api_name}", self.api_hash(api_name, claim))

    def set_api_response(self, api_name: str, claim: str, result: Any) -> bool:
        return self._set(f"api:{api_name}", self.api_hash(api_name, claim), result)

    def get_similarity_result(self, claim: str) -> dict[str, Any] | None:
        return self._get("similarity", self.claim_hash(claim))

    def set_similarity_result(self, claim: str, result: dict[str, Any]) -> bool:
        return self._set("similarity", self.claim_hash(claim), result)

    # ── Internal ─────────────────────────────────────────────

    def _get(self, namespace: str, digest: str) -> Any | None:
        key = self._key(namespace, digest)
        result = self._backend.get_json(key)
        if result is None:
            logger.debug("Cache miss: %s", key)
            return None
        logger.debug("Cache hit: %s", key)
        return result

    def _set(self, namespace: str, digest: str, value: Any) -> bool:
        key = self._key(namespace, digest)
        written = self._backend.set_json(key, value, self.ttl_seconds)
        if written:
            logger.debug("Cache write OK: %s", key)
        else:
            logger.warning("Cache write failed for key: %s", key)
        return written


_DEFAULT_MANAGER: CacheManager | None = None


def get_cache_manager() -> CacheManager:
    global _DEFAULT_MANAGER
    if _DEFAULT_MANAGER is None:
        _DEFAULT_MANAGER = CacheManager()
    return _DEFAULT_MANAGER
