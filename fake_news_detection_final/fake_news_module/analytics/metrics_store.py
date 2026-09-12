"""SQLite metrics storage for operational analytics."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from fake_news_module.config import ANALYTICS_DB_PATH


class MetricsStore:
    """SQLite-backed append-only metrics store."""

    def __init__(self, db_path: str = ANALYTICS_DB_PATH) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.initialize_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    claim TEXT NOT NULL,
                    final_decision TEXT NOT NULL,
                    confidence_value REAL NOT NULL,
                    score REAL NOT NULL,
                    pipeline_duration REAL NOT NULL,
                    api_duration REAL NOT NULL,
                    similarity_duration REAL NOT NULL,
                    roberta_duration REAL NOT NULL,
                    cache_hit INTEGER NOT NULL,
                    api_success_count INTEGER NOT NULL,
                    api_failure_count INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS claim_topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER NOT NULL,
                    topic TEXT NOT NULL,
                    FOREIGN KEY (analysis_id) REFERENCES analyses(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT NOT NULL,
                    namespace TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_events_type ON cache_events(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_claim_topics_topic ON claim_topics(topic)")

    def record_analysis(
        self,
        *,
        claim: str,
        final_decision: str,
        confidence_value: float,
        score: float,
        pipeline_duration: float,
        api_duration: float,
        similarity_duration: float,
        roberta_duration: float,
        cache_hit: bool,
        api_success_count: int,
        api_failure_count: int,
        topics: list[str],
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO analyses (
                    claim,
                    final_decision,
                    confidence_value,
                    score,
                    pipeline_duration,
                    api_duration,
                    similarity_duration,
                    roberta_duration,
                    cache_hit,
                    api_success_count,
                    api_failure_count
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    claim,
                    final_decision,
                    confidence_value,
                    score,
                    pipeline_duration,
                    api_duration,
                    similarity_duration,
                    roberta_duration,
                    int(cache_hit),
                    api_success_count,
                    api_failure_count,
                ),
            )
            analysis_id = int(cursor.lastrowid)
            conn.executemany(
                "INSERT INTO claim_topics (analysis_id, topic) VALUES (?, ?)",
                [(analysis_id, topic) for topic in topics],
            )
            return analysis_id

    def record_cache_event(self, event_type: str, namespace: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO cache_events (event_type, namespace) VALUES (?, ?)",
                (event_type, namespace),
            )

    def summary(self) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_analyses,
                    SUM(CASE WHEN final_decision = 'Real' THEN 1 ELSE 0 END) AS real_count,
                    SUM(CASE WHEN final_decision = 'Fake' THEN 1 ELSE 0 END) AS fake_count,
                    SUM(CASE WHEN final_decision = 'Uncertain' THEN 1 ELSE 0 END) AS uncertain_count,
                    AVG(confidence_value) AS average_confidence,
                    AVG(pipeline_duration) AS average_latency,
                    SUM(api_success_count) AS api_success_count,
                    SUM(api_failure_count) AS api_failure_count,
                    SUM(cache_hit) AS pipeline_cache_hits
                FROM analyses
                """
            ).fetchone()
            cache = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN event_type = 'hit' THEN 1 ELSE 0 END) AS hits,
                    SUM(CASE WHEN event_type = 'miss' THEN 1 ELSE 0 END) AS misses
                FROM cache_events
                """
            ).fetchone()
            topics = conn.execute(
                """
                SELECT topic, COUNT(*) AS count
                FROM claim_topics
                GROUP BY topic
                ORDER BY count DESC, topic ASC
                LIMIT 10
                """
            ).fetchall()
            claims = conn.execute(
                """
                SELECT claim, COUNT(*) AS count
                FROM analyses
                GROUP BY claim
                ORDER BY count DESC, claim ASC
                LIMIT 10
                """
            ).fetchall()

        total = int(row["total_analyses"] or 0)
        success = int(row["api_success_count"] or 0)
        failures = int(row["api_failure_count"] or 0)
        api_total = success + failures
        hits = int(cache["hits"] or 0)
        misses = int(cache["misses"] or 0)
        cache_total = hits + misses

        return {
            "total_analyses": total,
            "real_count": int(row["real_count"] or 0),
            "fake_count": int(row["fake_count"] or 0),
            "uncertain_count": int(row["uncertain_count"] or 0),
            "average_confidence": float(row["average_confidence"] or 0.0),
            "average_latency": float(row["average_latency"] or 0.0),
            "api_success_rate": (success / api_total) if api_total else 0.0,
            "api_failure_rate": (failures / api_total) if api_total else 0.0,
            "cache_hit_rate": (hits / cache_total) if cache_total else 0.0,
            "cache_miss_rate": (misses / cache_total) if cache_total else 0.0,
            "cache_hits": hits,
            "cache_misses": misses,
            "pipeline_cache_hits": int(row["pipeline_cache_hits"] or 0),
            "most_common_topics": [dict(item) for item in topics],
            "most_common_claims": [dict(item) for item in claims],
        }

    def performance(self) -> dict[str, Any]:
        with self._connect() as conn:
            averages = conn.execute(
                """
                SELECT
                    AVG(pipeline_duration) AS average_pipeline_duration,
                    AVG(api_duration) AS average_api_duration,
                    AVG(similarity_duration) AS average_similarity_duration,
                    AVG(roberta_duration) AS average_roberta_duration
                FROM analyses
                """
            ).fetchone()
            trend = conn.execute(
                """
                SELECT created_at, pipeline_duration, api_duration, similarity_duration, roberta_duration
                FROM analyses
                ORDER BY id DESC
                LIMIT 20
                """
            ).fetchall()

        return {
            "average_pipeline_duration": float(averages["average_pipeline_duration"] or 0.0),
            "average_api_duration": float(averages["average_api_duration"] or 0.0),
            "average_similarity_duration": float(averages["average_similarity_duration"] or 0.0),
            "average_roberta_duration": float(averages["average_roberta_duration"] or 0.0),
            "latency_trend": [dict(row) for row in reversed(trend)],
        }


_DEFAULT_STORE: MetricsStore | None = None


def get_metrics_store() -> MetricsStore:
    global _DEFAULT_STORE
    if _DEFAULT_STORE is None:
        _DEFAULT_STORE = MetricsStore()
    return _DEFAULT_STORE
