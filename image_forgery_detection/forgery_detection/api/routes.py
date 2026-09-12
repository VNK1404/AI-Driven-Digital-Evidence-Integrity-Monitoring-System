"""
forgery_detection/api/routes.py
================================
Flask Blueprint that adds a single AI-powered forgery detection endpoint
to any existing Flask application.

Integration
-----------
    # In your main app.py:
    from forgery_detection.api.routes import forgery_bp
    app.register_blueprint(forgery_bp)

Endpoint
--------
    POST /ai-detect
        Form-data:  file=<image file>   (JPG / PNG / TIFF / WEBP)
        Returns:    JSON { label, confidence, scores }

    GET  /ai-detect/health
        Returns:    API status + model info
"""

import io
from datetime import datetime

from flask import Blueprint, request, jsonify
from PIL import Image

from forgery_detection.core.predictor import ForgeryPredictor

# Lazy predictor — loaded on first request to avoid slowing down app startup
_predictor: ForgeryPredictor | None = None

ALLOWED_MIME = {"image/jpeg", "image/png", "image/tiff", "image/webp"}
ALLOWED_EXT  = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

forgery_bp = Blueprint("forgery_detection", __name__, url_prefix="/ai-detect")


def _get_predictor() -> ForgeryPredictor:
    """Return (and lazily initialise) the singleton predictor."""
    global _predictor
    if _predictor is None:
        _predictor = ForgeryPredictor()
    return _predictor


def _error(msg: str, code: int = 400):
    return jsonify({"success": False, "error": msg}), code


def _ok(data: dict, code: int = 200):
    return jsonify({"success": True, **data}), code


# ── Health ────────────────────────────────────────────────────────────────────

@forgery_bp.get("/health")
def health():
    """Check if the forgery detection service is ready."""
    return _ok({
        "service":     "Image Forgery Detection",
        "model":       "3-Stage Ensemble (ResNet50 + EfficientNet-B0 + Meta-Learner)",
        "classes":     ["authentic", "tampered"],
        "timestamp":   datetime.utcnow().isoformat(),
        "supported_formats": list(ALLOWED_EXT),
    })


# ── Predict ───────────────────────────────────────────────────────────────────

@forgery_bp.post("")          # POST /ai-detect
@forgery_bp.post("/")         # POST /ai-detect/
def predict():
    """
    Run forgery detection on an uploaded image.

    Request (multipart/form-data):
        file  — image file (JPG, PNG, TIFF, WEBP)

    Response:
        {
            "success":    true,
            "label":      "tampered" | "authentic",
            "confidence": 0.9231,
            "scores":     { "authentic": 0.0769, "tampered": 0.9231 },
            "timestamp":  "2026-08-11T04:11:00"
        }
    """
    if "file" not in request.files:
        return _error("No file uploaded. Send as multipart/form-data with key 'file'.")

    f = request.files["file"]

    if not f.filename:
        return _error("File has no name.")

    # Validate by extension (MIME types can be spoofed)
    ext = "." + f.filename.rsplit(".", 1)[-1].lower() if "." in f.filename else ""
    if ext not in ALLOWED_EXT:
        return _error(
            f"Unsupported file type '{ext}'. Accepted: {sorted(ALLOWED_EXT)}"
        )

    try:
        img_bytes = f.read()
        image = Image.open(io.BytesIO(img_bytes))
    except Exception as exc:
        return _error(f"Could not open image: {exc}")

    try:
        predictor = _get_predictor()
        result    = predictor.predict(image)
    except Exception as exc:
        return _error(f"Inference failed: {exc}", code=500)

    return _ok({
        "label":      result["label"],
        "confidence": result["confidence"],
        "scores":     result["scores"],
        "file_name":  f.filename,
        "timestamp":  datetime.utcnow().isoformat(),
    })
