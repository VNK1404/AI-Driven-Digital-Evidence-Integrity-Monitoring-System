"""
schemas/results.py
-------------------
Standard result schemas for every forensic module adapter.

Every adapter in integrations/ returns a ModuleResult.
The orchestrator assembles them into a ForensicReport.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def sanitize_json_obj(obj: Any) -> Any:
    """
    Recursively convert non-JSON-serializable objects (e.g. IFDRational, bytes,
    datetime, numpy scalars, Path objects) into standard JSON-serializable primitives.
    """
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize_json_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_json_obj(v) for v in obj]
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ModuleResult:
    """
    Standardised output from a single forensic module adapter.

    Attributes
    ----------
    evidence_id : str
        The evidence ID this result belongs to.
    analysis_type : str
        One of: blockchain, metadata_forensics, image_forgery,
        deepfake_detection, fake_news_detection.
    status : str
        'success' | 'failed' | 'skipped'
    result : dict
        Module-specific output payload.  Empty dict when skipped/failed.
    error : str | None
        Error message if status == 'failed'.
    timestamp : str
        UTC ISO-8601 timestamp of when this result was produced.
    duration_seconds : float
        Wall-clock time taken by this module.
    """

    evidence_id: str
    analysis_type: str
    status: str                          # success | failed | skipped
    result: dict = field(default_factory=dict)
    error: str | None = None
    timestamp: str = field(default_factory=_utc_now)
    duration_seconds: float = 0.0

    # Optional: reason for skipped status
    skip_reason: str | None = None

    def to_dict(self) -> dict:
        d = {
            "evidence_id":       self.evidence_id,
            "analysis_type":     self.analysis_type,
            "status":            self.status,
            "timestamp":         self.timestamp,
            "duration_seconds":  round(self.duration_seconds, 3),
            "result":            self.result,
            "error":             self.error,
        }
        if self.skip_reason:
            d["skip_reason"] = self.skip_reason
        return sanitize_json_obj(d)

    @classmethod
    def skipped(
        cls,
        evidence_id: str,
        analysis_type: str,
        reason: str,
    ) -> "ModuleResult":
        """Convenience constructor for a skipped result."""
        return cls(
            evidence_id=evidence_id,
            analysis_type=analysis_type,
            status="skipped",
            skip_reason=reason,
        )

    @classmethod
    def failed(
        cls,
        evidence_id: str,
        analysis_type: str,
        error: str,
        duration: float = 0.0,
    ) -> "ModuleResult":
        """Convenience constructor for a failed result."""
        return cls(
            evidence_id=evidence_id,
            analysis_type=analysis_type,
            status="failed",
            error=error,
            duration_seconds=duration,
        )


@dataclass
class ForensicReport:
    """
    The top-level report assembled by the orchestrator for one evidence item.

    Contains the EvidenceObject metadata + one ModuleResult per applicable module.
    """

    evidence: dict                           # EvidenceObject.to_dict()
    blockchain: ModuleResult | None = None
    metadata: ModuleResult | None = None
    image_forgery: ModuleResult | None = None
    deepfake: ModuleResult | None = None
    fake_news: ModuleResult | None = None
    overall_status: str = "completed"        # completed | partial | failed
    analysis_timestamp: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict:
        def _module_dict(m: ModuleResult | None) -> dict | None:
            return m.to_dict() if m is not None else None

        d = {
            "evidence_id":        self.evidence.get("evidence_id"),
            "file":               self.evidence,
            "blockchain":         _module_dict(self.blockchain),
            "metadata":           _module_dict(self.metadata),
            "image_forgery":      _module_dict(self.image_forgery),
            "deepfake":           _module_dict(self.deepfake),
            "fake_news":          _module_dict(self.fake_news),
            "overall_status":     self.overall_status,
            "analysis_timestamp": self.analysis_timestamp,
        }
        return sanitize_json_obj(d)
