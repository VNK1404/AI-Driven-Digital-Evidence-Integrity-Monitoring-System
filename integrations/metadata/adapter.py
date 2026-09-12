"""
integrations/metadata/adapter.py
----------------------------------
Adapter for the metadata_forensics module.

Key challenges handled here
---------------------------
1. metadata_pipeline.py uses bare imports (from extraction import ...) that
   assume metadata_forensics/ is on sys.path.  This adapter injects it.
2. metadata_pipeline.py imports extract_audio_metadata — this import may fail
   if mutagen is not installed.  The adapter gates audio files out before any
   call reaches the pipeline, so this never matters at runtime.
3. Returns a standardised ModuleResult regardless of pipeline outcome.

Supported file types: image, video, document
Rejected file types:  audio (returns status='skipped')
"""

import sys
import time
import logging
from pathlib import Path

from config.settings import METADATA_ROOT
from schemas.results import ModuleResult

logger = logging.getLogger(__name__)

_SUPPORTED_TYPES = {"image", "video", "document"}

# ── sys.path injection ─────────────────────────────────────────────────────────
_metadata_added = False


def _ensure_path():
    global _metadata_added
    if not _metadata_added:
        root = str(METADATA_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        # Also add pipeline/ so its bare imports resolve
        pipeline_dir = str(METADATA_ROOT / "pipeline")
        if pipeline_dir not in sys.path:
            sys.path.insert(0, pipeline_dir)
        _metadata_added = True


def _get_pipeline():
    _ensure_path()
    from pipeline.metadata_pipeline import run_metadata_pipeline  # type: ignore[import]
    return run_metadata_pipeline


# ── Public API ─────────────────────────────────────────────────────────────────

def analyze_metadata(
    evidence_id: str,
    file_path: str,
    file_type: str,
) -> ModuleResult:
    """
    Run the full metadata forensic pipeline on a file.

    Parameters
    ----------
    evidence_id : str
        The shared evidence ID.
    file_path : str
        Absolute path to the evidence file.
    file_type : str
        'image' | 'video' | 'document'  (audio is rejected)

    Returns
    -------
    ModuleResult with status 'success' | 'failed' | 'skipped'
    """
    # ── Guard: reject audio ────────────────────────────────────────────────────
    if file_type not in _SUPPORTED_TYPES:
        return ModuleResult.skipped(
            evidence_id=evidence_id,
            analysis_type="metadata_forensics",
            reason=f"file_type '{file_type}' is not supported (audio excluded)",
        )

    t0 = time.perf_counter()
    try:
        run_pipeline = _get_pipeline()

        path = Path(file_path)
        if not path.exists():
            return ModuleResult.failed(
                evidence_id=evidence_id,
                analysis_type="metadata_forensics",
                error=f"File not found: {file_path}",
                duration=time.perf_counter() - t0,
            )

        # Map integration file_type to what the pipeline expects
        # pipeline accepts: 'image', 'video', 'audio', 'document'
        pipeline_result = run_pipeline(
            file_path=str(file_path),
            file_type=file_type,
        )

        raw_meta = pipeline_result.get("metadata", {})
        if isinstance(raw_meta, dict) and "extraction_error" in raw_meta:
            return ModuleResult.failed(
                evidence_id=evidence_id,
                analysis_type="metadata_forensics",
                error=str(raw_meta["extraction_error"]),
                duration=time.perf_counter() - t0,
            )

        # Normalise result to a clean, serialisable dict
        result = {
            "metadata_score":  pipeline_result.get("metadata_score", -1),
            "flags":           pipeline_result.get("flags", []),
            "anomaly":         bool(pipeline_result.get("anomaly", False)),
            "anomaly_method":  "z-score reference profile",
            "timeline_flags":  pipeline_result.get("timeline_flags", []),
            "features":        pipeline_result.get("features", {}),
            "raw_metadata":    _truncate_metadata(pipeline_result.get("metadata", {})),
        }

        return ModuleResult(
            evidence_id=evidence_id,
            analysis_type="metadata_forensics",
            status="success",
            duration_seconds=time.perf_counter() - t0,
            result=result,
        )

    except Exception as exc:
        logger.error(
            "metadata.analyze_metadata failed for %s: %s",
            evidence_id, exc, exc_info=True,
        )
        return ModuleResult.failed(
            evidence_id=evidence_id,
            analysis_type="metadata_forensics",
            error=str(exc),
            duration=time.perf_counter() - t0,
        )


def _truncate_metadata(metadata: dict, max_str_len: int = 500) -> dict:
    """
    Truncate overly long string values in the raw metadata dict
    to keep JSON reports reasonably sized.
    """
    if not isinstance(metadata, dict):
        return {}
    result = {}
    for k, v in metadata.items():
        if isinstance(v, str) and len(v) > max_str_len:
            result[k] = v[:max_str_len] + "…"
        elif isinstance(v, dict):
            result[k] = _truncate_metadata(v, max_str_len)
        else:
            result[k] = v
    return result
