"""
forgery_detection/run_server.py
================================
Standalone FastAPI server — alternative to integrating into the Flask app.
Runs on port 8000.

Usage
-----
    # From the project root (TP industry/):
    python forgery_detection/run_server.py

    # Or double-click start.bat (Windows)
"""

import os
import sys

# Fix OpenMP duplicate-library error on Windows with Anaconda
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import asyncio
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from forgery_detection.core.predictor import ForgeryPredictor


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Image Forgery Detection API",
    description=(
        "3-Stage Ensemble (ResNet50 + EfficientNet-B0 + Logistic Regression "
        "Meta-Learner) for detecting authentic vs. tampered images."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # restrict in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Load models once at startup
predictor = ForgeryPredictor()


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    label:      str
    confidence: float
    scores:     dict[str, float]
    file_name:  str
    timestamp:  str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health():
    """Service health check."""
    return {
        "status":    "ok",
        "service":   "Image Forgery Detection API",
        "model":     "3-Stage Ensemble",
        "timestamp": datetime.utcnow().isoformat(),
        "docs":      "http://localhost:8000/docs",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    """
    Detect whether an uploaded image is authentic or tampered.

    - **file**: image file (JPEG, PNG, TIFF, WEBP)
    """
    allowed_mime = {"image/jpeg", "image/png", "image/tiff", "image/webp"}
    if file.content_type not in allowed_mime:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type '{file.content_type}'. Use: {sorted(allowed_mime)}",
        )

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not open image: {exc}")

    # Run inference in a thread-pool to keep the async loop unblocked
    result = await asyncio.to_thread(predictor.predict, image)

    return PredictionResponse(
        label      = result["label"],
        confidence = result["confidence"],
        scores     = result["scores"],
        file_name  = file.filename or "unknown",
        timestamp  = datetime.utcnow().isoformat(),
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  Image Forgery Detection — Standalone API Server")
    print("=" * 60)
    print("  URL   : http://localhost:8000")
    print("  Docs  : http://localhost:8000/docs")
    print("  POST  : http://localhost:8000/predict")
    print("=" * 60)

    uvicorn.run("forgery_detection.run_server:app", host="0.0.0.0", port=8000, reload=False)
