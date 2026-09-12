"""
config/settings.py
------------------
Central configuration for the AI-Driven Digital Evidence Integrity
Monitoring System — unified integration layer.

All machine-specific paths and secrets are read from environment
variables / .env file.  Never hardcode paths here.
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env (project-root .env takes priority over module-level ones) ────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent   # new_final/
load_dotenv(_PROJECT_ROOT / ".env", override=True)

# ── Module root paths (resolved once at import time) ──────────────────────────
BLOCKCHAIN_ROOT   = _PROJECT_ROOT / "blockchain_chain_of_custudy"
DEEPFAKE_ROOT     = _PROJECT_ROOT / "deepfake_detector"
FAKE_NEWS_ROOT    = _PROJECT_ROOT / "fake_news_detection_final"
IMAGE_FORGERY_ROOT = _PROJECT_ROOT / "image_forgery_detection"
METADATA_ROOT     = _PROJECT_ROOT / "metadata_forensics"

# ── Storage & Database paths ───────────────────────────────────────────────────
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = (
    os.environ.get("SUPABASE_KEY", "")
    or os.environ.get("SUPABASE_SECRET_KEY", "")
    or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
).strip()
SUPABASE_STORAGE_BUCKET = os.environ.get("SUPABASE_STORAGE_BUCKET", "evidence-files").strip()

def is_supabase_enabled() -> bool:
    """Return True if both SUPABASE_URL and SUPABASE_KEY are non-empty."""
    return bool(SUPABASE_URL and SUPABASE_KEY)

REPORTS_DIR = _PROJECT_ROOT / "reports"
LOGS_DIR    = _PROJECT_ROOT / "logs"
LOG_FILE    = LOGS_DIR / "forensic_system.log"

REPORTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ── API server ─────────────────────────────────────────────────────────────────
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8080"))
API_DEBUG = os.environ.get("API_DEBUG", "false").lower() == "true"

# ── Evidence ID prefix ─────────────────────────────────────────────────────────
EVIDENCE_ID_PREFIX = "EV-"

# ── Tesseract (system-level dependency, configurable via env) ──────────────────
TESSERACT_CMD = os.environ.get(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# ── Logging setup ──────────────────────────────────────────────────────────────
def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with both file and console handlers."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        handlers.append(logging.FileHandler(LOG_FILE, encoding="utf-8"))
    except OSError:
        pass   # If log dir is not writable, console only

    logging.basicConfig(level=log_level, format=fmt, handlers=handlers, force=True)


# ── Supported file extensions (audio is explicitly excluded) ───────────────────
IMAGE_EXTENSIONS   = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS   = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".flv"}
DOCUMENT_EXTENSIONS = {".pdf"}

SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | DOCUMENT_EXTENSIONS


def get_file_category(file_path: str | Path) -> str:
    """Return 'image', 'video', 'document', or 'unsupported'."""
    ext = Path(file_path).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in DOCUMENT_EXTENSIONS:
        return "document"
    return "unsupported"
