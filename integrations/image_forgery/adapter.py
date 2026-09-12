"""
integrations/image_forgery/adapter.py
---------------------------------------
Adapter for the image_forgery_detection module.

Key design decisions
---------------------
1. ForgeryPredictor is a singleton — models (ResNet50 + EfficientNet-B0 +
   LogisticRegression meta-learner) are loaded ONCE on first call and reused.
2. We inject image_forgery_detection/ onto sys.path so that
   `from forgery_detection.core.predictor import ForgeryPredictor` resolves.
3. This adapter ONLY runs on IMAGE files.  Any other file type is returned
   as status='skipped'.
4. The adapter normalises the predictor output into the standard ModuleResult.

Predictor output
----------------
    {
        "label":      "authentic" | "tampered",
        "confidence": float (0.0–1.0),
        "scores":     {"authentic": float, "tampered": float},
    }
"""

import sys
import time
import logging
from pathlib import Path

from config.settings import IMAGE_FORGERY_ROOT
from schemas.results import ModuleResult

logger = logging.getLogger(__name__)

# ── sys.path injection ─────────────────────────────────────────────────────────
_forgery_added = False
_predictor_instance = None


def _ensure_path():
    global _forgery_added
    if not _forgery_added:
        root = str(IMAGE_FORGERY_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        _forgery_added = True


def _get_predictor():
    """Return the ForgeryPredictor singleton.  Models are loaded on first call."""
    global _predictor_instance
    if _predictor_instance is None:
        _ensure_path()
        from forgery_detection.core.predictor import ForgeryPredictor  # type: ignore[import]
        logger.info("Loading image forgery models (ResNet50 + EfficientNet-B0)…")
        _predictor_instance = ForgeryPredictor()
        logger.info("Image forgery models loaded.")
    return _predictor_instance


# ── Public API ─────────────────────────────────────────────────────────────────

def analyze_image_forgery(
    evidence_id: str,
    file_path: str,
    file_type: str = "image",
) -> ModuleResult:
    """
    Run the 3-stage ensemble forgery detector on an image file.

    Parameters
    ----------
    evidence_id : str
        Shared evidence ID.
    file_path : str
        Absolute path to the image file.
    file_type : str
        Must be 'image'.  Any other value returns status='skipped'.

    Returns
    -------
    ModuleResult
        result keys: prediction, confidence, scores, model
    """
    if file_type != "image":
        return ModuleResult.skipped(
            evidence_id=evidence_id,
            analysis_type="image_forgery",
            reason=f"image_forgery only runs on images, got file_type='{file_type}'",
        )

    t0 = time.perf_counter()
    try:
        predictor = _get_predictor()
        raw = predictor.predict_from_path(str(file_path))

        label = raw.get("label", "unknown").upper()          # AUTHENTIC or TAMPERED
        confidence = float(raw.get("confidence", 0.0))
        scores = raw.get("scores", {})

        return ModuleResult(
            evidence_id=evidence_id,
            analysis_type="image_forgery",
            status="success",
            duration_seconds=time.perf_counter() - t0,
            result={
                "prediction":  label,
                "confidence":  round(confidence, 4),
                "scores": {
                    "authentic": round(float(scores.get("authentic", 0.0)), 4),
                    "tampered":  round(float(scores.get("tampered",  0.0)), 4),
                },
                "model": "ResNet50 + EfficientNet-B0 + Logistic Regression",
            },
        )

    except Exception as exc:
        logger.error(
            "image_forgery.analyze_image_forgery failed for %s: %s",
            evidence_id, exc, exc_info=True,
        )
        return ModuleResult.failed(
            evidence_id=evidence_id,
            analysis_type="image_forgery",
            error=str(exc),
            duration=time.perf_counter() - t0,
        )
