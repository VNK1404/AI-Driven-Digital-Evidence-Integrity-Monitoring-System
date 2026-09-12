"""
forgery_detection/core/predictor.py
=====================================
ForgeryPredictor — the single public interface for running inference.

Singleton-safe: models are loaded once on first instantiation and reused.

Usage
-----
from forgery_detection.core.predictor import ForgeryPredictor
from PIL import Image

predictor = ForgeryPredictor()                   # loads models once
result    = predictor.predict(Image.open("x.jpg"))

# result:
# {
#   "label":      "tampered" | "authentic",
#   "confidence": 0.9231,
#   "scores":     { "authentic": 0.0769, "tampered": 0.9231 }
# }

# You can also pass a file path directly:
result = predictor.predict_from_path("image.jpg")
"""

import os

# Fix OpenMP duplicate-library error that occurs on Windows with Anaconda + PyTorch
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from forgery_detection.core.models import load_models

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ["authentic", "tampered"]   # must match train_dataset.class_to_idx

# ImageNet normalisation — identical to training time
_INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ── Singleton state ───────────────────────────────────────────────────────────
_instance = None


class ForgeryPredictor:
    """
    Wraps the 3-stage ensemble and exposes `predict(image)`.

    Thread-safe for read-only inference calls (no model state changes).
    Call `ForgeryPredictor()` anywhere — the models are loaded only once
    globally.
    """

    def __new__(cls, **kwargs):
        """Return the existing singleton or create a new one."""
        global _instance
        if _instance is None:
            _instance = super().__new__(cls)
            _instance._initialised = False
        return _instance

    def __init__(
        self,
        resnet_path: str | None = None,
        effnet_path: str | None = None,
        meta_path:   str | None = None,
        device: torch.device | None = None,
    ):
        if self._initialised:
            return  # already loaded, skip

        load_kwargs = {}
        if resnet_path: load_kwargs["resnet_path"] = resnet_path
        if effnet_path: load_kwargs["effnet_path"] = effnet_path
        if meta_path:   load_kwargs["meta_path"]   = meta_path
        if device:      load_kwargs["device"]       = device

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.m1, self.m2, self.meta = load_models(**load_kwargs, device=self.device)
        self._initialised = True

    # ── Core inference ────────────────────────────────────────────────────────

    def predict(self, image: Image.Image) -> dict:
        """
        Run the full 3-stage ensemble on a PIL image.

        Parameters
        ----------
        image : PIL.Image.Image  (any mode — auto-converted to RGB)

        Returns
        -------
        dict with keys: label, confidence, scores
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = _INFER_TRANSFORM(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit1 = self.m1(tensor).cpu().numpy()   # (1, 2)
            logit2 = self.m2(tensor).cpu().numpy()   # (1, 2)

        stacked = np.hstack([logit1, logit2])         # (1, 4)

        probs    = self.meta.predict_proba(stacked)[0]   # (2,)
        pred_idx = int(self.meta.predict(stacked)[0])

        scores = {CLASS_NAMES[i]: round(float(probs[i]), 4)
                  for i in range(len(CLASS_NAMES))}

        return {
            "label":      CLASS_NAMES[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
            "scores":     scores,
        }

    def predict_from_path(self, file_path: str) -> dict:
        """
        Convenience wrapper — load an image from disk and predict.

        Parameters
        ----------
        file_path : absolute or relative path to an image file

        Returns
        -------
        dict with keys: label, confidence, scores, file_path
        """
        image = Image.open(file_path)
        result = self.predict(image)
        result["file_path"] = file_path
        return result
