"""
integrations/deepfake/adapter.py
----------------------------------
Adapter for the deepfake_detector module.

Key design decisions
---------------------
1. We bypass the Django web app and the CLI argparse main() entirely.
   Instead, we import video_to_tensor() and predict() from predict.py
   directly after injecting deepfake_detector/ onto sys.path.
2. The model (EfficientNet-B3 + BiLSTM + Attention, 66 MB) is loaded
   ONCE on first call using a module-level singleton.
3. predict.py calls sys.exit() on fatal errors inside its helper functions.
   The adapter catches SystemExit to prevent it from crashing the server.
4. This adapter ONLY runs on VIDEO files.  Any other file type is
   returned as status='skipped'.

Model output
------------
    label      : "REAL" | "FAKE"
    confidence : float (0.0–100.0)
    real_prob  : float
    fake_prob  : float
"""

import sys
import time
import logging
from pathlib import Path

from config.settings import DEEPFAKE_ROOT
from schemas.results import ModuleResult

logger = logging.getLogger(__name__)

# ── sys.path injection ─────────────────────────────────────────────────────────
_deepfake_added = False
_deepfake_model = None
_deepfake_device = None


def _ensure_path():
    global _deepfake_added
    if not _deepfake_added:
        root = str(DEEPFAKE_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        _deepfake_added = True


def _get_model_and_device():
    """
    Load the deepfake model singleton.  Called once; subsequent calls are no-ops.
    Returns (model, device).
    """
    global _deepfake_model, _deepfake_device
    if _deepfake_model is None:
        _ensure_path()
        import torch
        from predict import load_model  # type: ignore[import]
        import dataset_config as cfg    # type: ignore[import]

        if torch.cuda.is_available():
            try:
                torch.zeros(1, device="cuda")
                _deepfake_device = torch.device("cuda")
            except RuntimeError:
                _deepfake_device = torch.device("cpu")
        else:
            _deepfake_device = torch.device("cpu")

        logger.info(
            "Loading deepfake model from %s on %s…",
            cfg.BEST_MODEL_PATH, _deepfake_device
        )
        _deepfake_model = load_model(cfg.BEST_MODEL_PATH, _deepfake_device)
        logger.info("Deepfake model loaded.")

    return _deepfake_model, _deepfake_device


def _get_inference_fns():
    _ensure_path()
    from predict import video_to_tensor, predict  # type: ignore[import]
    import dataset_config as cfg                  # type: ignore[import]
    return video_to_tensor, predict, cfg


# ── Public API ─────────────────────────────────────────────────────────────────

def analyze_deepfake(
    evidence_id: str,
    file_path: str,
    file_type: str = "video",
) -> ModuleResult:
    """
    Run the deepfake detection model on a video file.

    Parameters
    ----------
    evidence_id : str
        Shared evidence ID.
    file_path : str
        Absolute path to the video file.
    file_type : str
        Must be 'video'.  Any other value returns status='skipped'.

    Returns
    -------
    ModuleResult
        result keys: prediction, confidence_pct, real_prob, fake_prob,
                     frames_analyzed, model, device
    """
    if file_type != "video":
        return ModuleResult.skipped(
            evidence_id=evidence_id,
            analysis_type="deepfake_detection",
            reason=f"deepfake only runs on videos, got file_type='{file_type}'",
        )

    t0 = time.perf_counter()
    try:
        model, device = _get_model_and_device()
        video_to_tensor, predict_fn, cfg = _get_inference_fns()

        logger.info("Processing video for deepfake analysis: %s", file_path)
        tensor = video_to_tensor(str(file_path), sequence_length=cfg.SEQUENCE_LENGTH)
        label, confidence, real_prob, fake_prob = predict_fn(model, tensor, device)

        return ModuleResult(
            evidence_id=evidence_id,
            analysis_type="deepfake_detection",
            status="success",
            duration_seconds=time.perf_counter() - t0,
            result={
                "prediction":     label,           # "REAL" | "FAKE"
                "confidence_pct": round(confidence, 2),
                "real_prob":      round(real_prob, 2),
                "fake_prob":      round(fake_prob, 2),
                "frames_analyzed": cfg.SEQUENCE_LENGTH,
                "model": "EfficientNet-B3 + BiLSTM + Attention",
                "device": str(device),
            },
        )

    except SystemExit as exc:
        # predict.py uses sys.exit() internally — intercept it
        logger.error(
            "deepfake.analyze_deepfake caught sys.exit(%s) for %s",
            exc.code, evidence_id,
        )
        return ModuleResult.failed(
            evidence_id=evidence_id,
            analysis_type="deepfake_detection",
            error=f"deepfake module exited with code {exc.code}",
            duration=time.perf_counter() - t0,
        )
    except Exception as exc:
        logger.error(
            "deepfake.analyze_deepfake failed for %s: %s",
            evidence_id, exc, exc_info=True,
        )
        return ModuleResult.failed(
            evidence_id=evidence_id,
            analysis_type="deepfake_detection",
            error=str(exc),
            duration=time.perf_counter() - t0,
        )
