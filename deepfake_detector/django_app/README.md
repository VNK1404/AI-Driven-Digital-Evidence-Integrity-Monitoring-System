# 🛡️ DeepShield — Deepfake Detection Web App

A Django-based web interface for the **EfficientNet-B3 + BiLSTM + Temporal Attention** deepfake detection model.

Upload any video → AI extracts frames → detects manipulated faces → verdict: **REAL** or **FAKE**.

---

## 📁 Directory Structure

```
django_app/
│
├── manage.py                        # Django management script
├── requirements.txt                 # Python dependencies
│
├── project_settings/                # Django project config
│   ├── settings.py                  # All settings (paths, DB, static, media)
│   ├── urls.py                      # Root URL config
│   └── wsgi.py
│
├── ml_app/                          # Core detection app
│   ├── views.py                     # ← Inference logic lives here
│   ├── forms.py                     # Video upload form
│   ├── urls.py                      # App-level URL routes
│   └── templates/
│       ├── base.html                # Navbar + footer shell
│       ├── index.html               # Home / upload page
│       ├── predict.html             # Results page
│       ├── about.html               # About page
│       └── error.html               # Graceful error display
│
├── static/
│   └── css/style.css                # Premium dark-themed stylesheet
│
├── models/                          # ← Place best_model_ffpp.pth HERE
│   └── README.txt
│
├── uploaded_videos/                 # Temporary storage for uploaded videos
└── uploaded_images/                 # Temporary storage for extracted frames
```

---

## ⚙️ Setup & Running

### Prerequisites
- Python 3.8+
- The project's `model.py` and `dataset_config.py` must be accessible from the **parent directory** of `django_app/`
- Your trained model file: `best_model_ffpp.pth`

### Step 1 — Install Dependencies

```bash
# Install PyTorch with CUDA first (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Then install remaining requirements
pip install -r requirements.txt
```

### Step 2 — Place Your Trained Model

Copy your trained model file into the `models/` folder:

```
django_app/
└── models/
    └── best_model_ffpp.pth   ← place it here
```

> The model is loaded automatically when the first prediction request is made and **cached** in memory for all subsequent requests.

### Step 3 — Run Database Migrations

```bash
python manage.py migrate
```

### Step 4 — Run the Development Server

```bash
python manage.py runserver
```

Open your browser and go to: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🌐 Pages

| URL | Description |
|---|---|
| `/` | Home page — drag & drop video upload |
| `/predict/` | Results page — shows REAL/FAKE verdict, confidence, frame previews |
| `/about/` | About page — architecture and model metrics |

---

## 🔬 How Inference Works

1. User uploads a video via the web form.
2. The video is saved temporarily to `uploaded_videos/`.
3. **15 evenly-spaced frames** are extracted from the video.
4. Each frame goes through **OpenCV Haar Cascade** face detection and cropping.
5. All frames are stacked into a tensor `(1, 15, 3, 224, 224)` and passed to the model.
6. **EfficientNet-B3** extracts per-frame spatial features.
7. **BiLSTM + Temporal Attention** models how features change across frames.
8. A **softmax** score gives the REAL/FAKE confidence percentage.
9. The result is displayed on the results page with frame previews.

---

## ⚠️ Important Notes

- The `models/` folder and `uploaded_videos/` are excluded from git (see `.gitignore`).
- This app **does not require `face_recognition`** — it uses OpenCV Haar Cascade, which is simpler to install.
- For production deployment, change `DEBUG = False` and set a proper `SECRET_KEY` in `settings.py`.
