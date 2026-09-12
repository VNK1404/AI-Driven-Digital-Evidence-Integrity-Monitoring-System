"""Analytics orchestration helpers for pipeline and dashboard code."""

from __future__ import annotations

import logging
import re
from typing import Any

from fake_news_module.analytics.metrics_store import MetricsStore, get_metrics_store

logger = logging.getLogger(__name__)

_STOPWORDS = {
    "about", "after", "again", "against", "also", "and", "are", "because", "been",
    "before", "being", "between", "claim", "from", "have", "into", "that", "the",
    "their", "there", "this", "through", "with", "would",
}


class AnalyticsService:
    def __init__(self, store: MetricsStore | None = None) -> None:
        self.store = store or get_metrics_store()

    def record_analysis(
        self,
        result: dict[str, Any],
        *,
        pipeline_duration: float,
        stage_timings: dict[str, float] | None = None,
        cache_hit: bool = False,
    ) -> None:
        stage_timings = stage_timings or {}
        api_results = result.get("api_results", {})
        success_count, failure_count = self._api_success_failure_counts(api_results)
        try:
            self.store.record_analysis(
                claim=result.get("claim", ""),
                final_decision=result.get("final_decision", "Uncertain"),
                confidence_value=self._confidence_value(result),
                score=float(result.get("score", 0.0) or 0.0),
                pipeline_duration=float(pipeline_duration),
                api_duration=float(stage_timings.get("external_apis", 0.0)),
                similarity_duration=float(stage_timings.get("similarity", 0.0)),
                roberta_duration=float(stage_timings.get("roberta", 0.0)),
                cache_hit=cache_hit,
                api_success_count=success_count,
                api_failure_count=failure_count,
                topics=self._topics(result.get("claim", "")),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Analytics recording failed: %s", exc)

    def summary(self) -> dict[str, Any]:
        return self.store.summary()

    def performance(self) -> dict[str, Any]:
        return self.store.performance()

    @staticmethod
    def _confidence_value(result: dict[str, Any]) -> float:
        confidence = result.get("confidence")
        if isinstance(confidence, (int, float)):
            return max(0.0, min(1.0, float(confidence)))
        mapping = {"high": 1.0, "medium": 0.6, "low": 0.3}
        return mapping.get(str(confidence or "").lower(), abs(float(result.get("score", 0.0) or 0.0)))

    @staticmethod
    def _api_success_failure_counts(api_results: dict[str, Any]) -> tuple[int, int]:
        success = 0
        failure = 0
        for value in api_results.values():
            label = str(value or "Unknown").lower()
            if label == "unknown":
                failure += 1
            else:
                success += 1
        return success, failure

    @staticmethod
    def _topics(claim: str, max_topics: int = 6) -> list[str]:
        words = re.findall(r"[a-zA-Z][a-zA-Z]{3,}", claim.lower())
        topics: list[str] = []
        for word in words:
            if word in _STOPWORDS or word in topics:
                continue
            topics.append(word)
            if len(topics) >= max_topics:
                break
        return topics


_DEFAULT_SERVICE: AnalyticsService | None = None


def get_analytics_service() -> AnalyticsService:
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = AnalyticsService()
    return _DEFAULT_SERVICE
