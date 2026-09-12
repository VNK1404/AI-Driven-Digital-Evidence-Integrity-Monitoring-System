# 🔍 Image Forgery Detection — Integration Package

A plug-and-play Python package that wraps a **3-stage ensemble model** (ResNet50 + EfficientNet-B0 + Logistic Regression Meta-Learner) for detecting authentic vs. tampered images.

---

## 📁 Folder Structure

```
forgery_detection/
├── core/
│   ├── models.py        # loads .pth weights + builds model architectures
│   └── predictor.py     # ForgeryPredictor class (the main interface)
├── api/
│   └── routes.py        # Flask Blueprint  →  POST /ai-detect
├── run_server.py        # standalone FastAPI server (port 8000)
├── start.bat            # one-click Windows startup
├── requirements.txt
└── README.md
```

---

## ⚙️ Model Weights Required

Place these files in the **project root** (`TP industry/`):

| File | Stage |
|---|---|
| `best_resnet50.pth` | Stage 1 — ResNet50 |
| `best_efficientnet.pth` | Stage 2 — EfficientNet-B0 |
| `meta_learner.pkl` | Stage 3 — Logistic Regression |

---

## 🚀 Option 1: Standalone FastAPI Server

```powershell
# From the project root:
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python forgery_detection/run_server.py
```

Or just **double-click `start.bat`** on Windows.

**Endpoints:**

| Method | URL | Description |
|---|---|---|
| `GET` | `http://localhost:8000/` | Health check |
| `POST` | `http://localhost:8000/predict` | Run inference |
| `GET` | `http://localhost:8000/docs` | Swagger UI |

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@your_image.jpg"
```

---

## 🔗 Option 2: Integrate into Existing Flask App

The `routes.py` registers a Blueprint that adds `/ai-detect` to any Flask app.
This is **already done** in `parth/forensic_system/forensic_system/api/app.py`.

**Endpoints added to the forensic app:**

| Method | URL | Description |
|---|---|---|
| `GET` | `/ai-detect/health` | Forgery service health |
| `POST` | `/ai-detect` | Run AI forgery detection |

---

## 🐍 Use as a Python Library

```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows only

from forgery_detection.core.predictor import ForgeryPredictor
from PIL import Image

# Load once (singleton — subsequent calls reuse loaded models)
predictor = ForgeryPredictor()

# Predict from PIL Image
result = predictor.predict(Image.open("test.jpg"))
print(result)
# {'label': 'tampered', 'confidence': 0.9231, 'scores': {'authentic': 0.0769, 'tampered': 0.9231}}

# Or predict directly from a file path
result = predictor.predict_from_path("test.jpg")
```

---

## 📦 Install Dependencies

```powershell
pip install -r forgery_detection/requirements.txt

# For PyTorch CPU (recommended for Windows without GPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## ⚠️ Windows / Anaconda Note

If you see `OMP: Error #15` on startup, set this environment variable:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'        # temporary (current session)

# Permanent fix:
[System.Environment]::SetEnvironmentVariable('KMP_DUPLICATE_LIB_OK','TRUE','User')
```

The `predictor.py` sets this automatically, but it must be set **before** any torch import.

---

## 📤 API Response Format

```json
{
  "success": true,
  "label": "tampered",
  "confidence": 0.9231,
  "scores": {
    "authentic": 0.0769,
    "tampered": 0.9231
  },
  "file_name": "evidence_photo.jpg",
  "timestamp": "2026-08-11T04:11:00"
}
```
