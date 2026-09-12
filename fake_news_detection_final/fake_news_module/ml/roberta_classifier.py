"""
fake_news_module/ml/roberta_classifier.py
=========================================
RoBERTa-based binary fake-news classifier.

Architecture
------------
    Input text
        └─► RoBERTa tokenizer (roberta-base, max_length=512)
                └─► RoBERTa encoder  (768-dim CLS token)
                        └─► Linear classifier head (768 → 2)
                                └─► Softmax → {FAKE: 0, REAL: 1}

The module implements a **lazy singleton** pattern:
  • The model is loaded only once on first call to `predict()`.
  • Thread-safe via a `threading.Lock`.
  • Gracefully degrades to `{"label": "Unknown", "confidence": 0.0}` when
    weights are unavailable (e.g., model not yet fine-tuned / downloaded).

Usage
-----
    from fake_news_module.ml.roberta_classifier import predict

    result = predict("NASA confirms life on Mars")
    # → {"label": "Real", "confidence": 0.87}
"""

import logging
import threading
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

# Label mapping: model output index → human-readable label
LABEL_MAP: Dict[int, str] = {0: "Fake", 1: "Real"}

# Maximum token length supported by roberta-base
MAX_TOKEN_LENGTH: int = 512

# Default pretrained backbone (used when no fine-tuned weights exist)
PRETRAINED_BASE: str = "roberta-base"

# ──────────────────────────────────────────────
# Lazy-loading singleton state
# ──────────────────────────────────────────────
_lock        = threading.Lock()
_tokenizer   = None   # transformers.AutoTokenizer
_model       = None   # transformers.AutoModelForSequenceClassification
_device      = None   # "cuda" or "cpu"
_load_failed = False  # set to True after a failed load; prevents endless retries


def _import_torch():
    """Lazily import PyTorch to avoid hard dependency at import time."""
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is not installed. Run: pip install torch"
        ) from exc


def _import_transformers():
    """Lazily import HuggingFace transformers."""
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        return AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as exc:
        raise ImportError(
            "transformers is not installed. Run: pip install transformers"
        ) from exc


def load_model(model_dir: Optional[str] = None) -> bool:
    """
    Load (or lazy-initialize) the RoBERTa classifier.

    Tries in order:
        1. Fine-tuned weights from `model_dir` (if provided and exists).
        2. Fine-tuned weights from default saved_model/ path.
        3. Pretrained `roberta-base` backbone (no classification head weights).
           In this case the model can still run inference but will give random
           class predictions until fine-tuned.

    Args:
        model_dir: Optional explicit path to a fine-tuned model directory.

    Returns:
        True  — model loaded successfully.
        False — failed to load (import error or IO error).
    """
    global _tokenizer, _model, _device, _load_failed

    with _lock:
        # Already loaded — skip
        if _model is not None:
            return True

        # Previous attempt failed — don't retry (model files are unlikely to
        # have changed since the last attempt in the same process)
        if _load_failed:
            logger.debug("RoBERTa: skipping reload after previous failure.")
            return False

        try:
            torch = _import_torch()
            AutoTokenizer, AutoModelForSequenceClassification = _import_transformers()

            # Determine device
            _device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("RoBERTa: using device '%s'", _device)

            # Resolve model path priority
            default_saved = Path(__file__).parent / "saved_model"
            candidate_dirs = []
            if model_dir:
                candidate_dirs.append(Path(model_dir))
            candidate_dirs.append(default_saved)

            chosen_path: Optional[Path] = None
            for p in candidate_dirs:
                if p.exists() and any(p.iterdir()):
                    chosen_path = p
                    logger.info("RoBERTa: loading fine-tuned model from '%s'", p)
                    break

            if chosen_path is None:
                # Fall back to pretrained backbone
                logger.warning(
                    "RoBERTa: no fine-tuned weights found. "
                    "Loading pretrained '%s' backbone. "
                    "Run train.py to fine-tune on your dataset.",
                    PRETRAINED_BASE,
                )
                chosen_path = PRETRAINED_BASE  # type: ignore[assignment]

            _tokenizer = AutoTokenizer.from_pretrained(str(chosen_path))
            _model = AutoModelForSequenceClassification.from_pretrained(
                str(chosen_path),
                num_labels=2,
            )
            _model.to(_device)
            _model.eval()

            logger.info("RoBERTa: model loaded successfully (num_labels=2, device=%s).", _device)
            return True

        except Exception as exc:  # noqa: BLE001
            _load_failed = True
            logger.error("RoBERTa: failed to load model — %s", exc)
            return False


def predict(text: str, model_dir: Optional[str] = None) -> Dict[str, object]:
    """
    Run inference on a single text string.

    Args:
        text:      Input claim / news text (will be truncated to MAX_TOKEN_LENGTH).
        model_dir: Optional override for the model directory (passed to load_model).

    Returns:
        A dict with:
            {
                "label":      "Real" | "Fake" | "Unknown",
                "confidence": float in [0.0, 1.0],
                "fake_prob":  float,   # probability of class FAKE
                "real_prob":  float,   # probability of class REAL
            }
        Returns Unknown on any error.
    """
    _UNKNOWN = {"label": "Unknown", "confidence": 0.0, "fake_prob": 0.0, "real_prob": 0.0}

    if not text or not text.strip():
        logger.warning("RoBERTa predict: empty text received.")
        return _UNKNOWN

    # Ensure model is loaded
    if _model is None:
        if not load_model(model_dir):
            return _UNKNOWN

    try:
        torch = _import_torch()

        # Tokenise
        inputs = _tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            padding="max_length",
        )
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        # Forward pass (no gradient needed)
        with torch.no_grad():
            outputs = _model(**inputs)
            logits = outputs.logits                  # shape: (1, 2)
            probs = torch.softmax(logits, dim=-1)    # shape: (1, 2)
            fake_prob = float(probs[0][0].item())
            real_prob = float(probs[0][1].item())

        # Predicted class
        pred_idx = int(torch.argmax(probs, dim=-1).item())
        label = LABEL_MAP[pred_idx]
        confidence = max(fake_prob, real_prob)

        logger.info(
            "RoBERTa predict: label='%s' confidence=%.4f "
            "(fake=%.4f real=%.4f)",
            label, confidence, fake_prob, real_prob,
        )

        return {
            "label":      label,
            "confidence": round(confidence, 4),
            "fake_prob":  round(fake_prob, 4),
            "real_prob":  round(real_prob, 4),
        }

    except Exception as exc:  # noqa: BLE001
        logger.error("RoBERTa predict: inference failed — %s", exc)
        return _UNKNOWN


def warm_up(model_dir=None):
    """
    Pre-load the model in the calling thread (intended for the main thread).

    Call once at application start-up before ThreadPoolExecutor workers
    start to avoid a Windows race condition where the first HuggingFace
    safetensors load inside a non-main thread can intermittently fail.

    Returns:
        True  -- model is ready.
        False -- model failed to load (see ERROR log).
    """
    logger.info("RoBERTa warm_up: pre-loading model ...")
    return load_model(model_dir)
