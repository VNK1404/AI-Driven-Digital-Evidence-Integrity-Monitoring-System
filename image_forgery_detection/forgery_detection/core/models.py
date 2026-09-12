"""
forgery_detection/core/models.py
=================================
Loads the 3-stage ensemble weights from disk — completely decoupled from
train_ensemble.py so this package can be used standalone anywhere.

Usage
-----
from forgery_detection.core.models import load_models
m1, m2, meta = load_models()          # defaults to project root weight paths
m1, m2, meta = load_models(           # custom paths
    resnet_path   = "path/to/best_resnet50.pth",
    effnet_path   = "path/to/best_efficientnet.pth",
    meta_path     = "path/to/meta_learner.pkl",
)
"""

import os
import torch
import torch.nn as nn
import joblib
from torchvision import models

# ── Default weight paths (relative to this file's package root) ───────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT   = os.path.dirname(_HERE)            # forgery_detection/
_PROJ_ROOT  = os.path.dirname(_PKG_ROOT)        # TP industry/ (project root)

def _resolve_weight_path(filename: str) -> str:
    # 1. Look inside the package root (forgery_detection/)
    pkg_path = os.path.join(_PKG_ROOT, filename)
    if os.path.isfile(pkg_path):
        return pkg_path
    # 2. Fallback to parent workspace folder (TP industry/)
    return os.path.join(_PROJ_ROOT, filename)

DEFAULT_RESNET_PATH = _resolve_weight_path("best_resnet50.pth")
DEFAULT_EFFNET_PATH = _resolve_weight_path("best_efficientnet.pth")
DEFAULT_META_PATH   = _resolve_weight_path("meta_learner.pkl")

NUM_CLASSES = 2


# ── Architecture builders (mirrors train_ensemble.py exactly) ─────────────────

def _build_resnet50(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = models.resnet50(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, num_classes),
    )
    return model


def _build_efficientnet(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.features[6:].parameters():
        param.requires_grad = True
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model


# ── Public loader ─────────────────────────────────────────────────────────────

def load_models(
    resnet_path: str = DEFAULT_RESNET_PATH,
    effnet_path: str = DEFAULT_EFFNET_PATH,
    meta_path:   str = DEFAULT_META_PATH,
    device: torch.device | None = None,
):
    """
    Load and return the three-stage ensemble.

    Parameters
    ----------
    resnet_path : path to best_resnet50.pth
    effnet_path : path to best_efficientnet.pth
    meta_path   : path to meta_learner.pkl
    device      : torch device (auto-detected if None)

    Returns
    -------
    (m1, m2, meta)
        m1   — ResNet50 (Stage 1), eval mode
        m2   — EfficientNet-B0 (Stage 2), eval mode
        meta — sklearn LogisticRegression (Stage 3)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validate files exist early for clearer error messages
    for label, path in [
        ("ResNet50 weights",       resnet_path),
        ("EfficientNet weights",   effnet_path),
        ("Meta-learner pickle",    meta_path),
    ]:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{label} not found at:\n  {path}\n"
                "Set the correct path via load_models(resnet_path=..., ...)"
            )

    print(f"[ForgeryDetection] Loading models on {device} ...")

    m1 = _build_resnet50(NUM_CLASSES)
    m1.load_state_dict(torch.load(resnet_path, map_location=device))
    m1.to(device).eval()

    m2 = _build_efficientnet(NUM_CLASSES)
    m2.load_state_dict(torch.load(effnet_path, map_location=device))
    m2.to(device).eval()

    meta = joblib.load(meta_path)
    if not hasattr(meta, "multi_class"):
        meta.multi_class = "auto"

    print(f"[ForgeryDetection] All 3 stages loaded successfully.")
    return m1, m2, meta
