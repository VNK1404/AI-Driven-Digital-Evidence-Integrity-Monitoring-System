"""
ocr/image_ocr.py — Image OCR using pytesseract + OpenCV
AI-Driven Digital Evidence Integrity Monitoring System
"""

import logging
import pytesseract
import cv2
import numpy as np
from pathlib import Path

from fake_news_module.config import TESSERACT_CMD

logger = logging.getLogger(__name__)

# Set tesseract binary path (Windows)
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def _preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Apply preprocessing pipeline to improve OCR accuracy:
    1. Convert to grayscale
    2. Apply bilateral filter (noise removal while preserving edges)
    3. Apply adaptive thresholding (handles uneven lighting)
    4. Deskew if needed (basic morphological clean-up)
    """
    # Step 1: Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Noise removal with bilateral filter
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 3: Adaptive thresholding for non-uniform illumination
    thresh = cv2.adaptiveThreshold(
        denoised,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,
        C=2,
    )

    # Step 4: Morphological opening to remove small noise dots
    kernel = np.ones((1, 1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return cleaned


def _scale_image_if_small(image: np.ndarray, min_height: int = 600) -> np.ndarray:
    """
    Upscale small images — Tesseract performs poorly on images < 300 DPI.
    """
    h, w = image.shape[:2]
    if h < min_height:
        scale_factor = min_height / h
        new_w = int(w * scale_factor)
        image = cv2.resize(image, (new_w, min_height), interpolation=cv2.INTER_CUBIC)
        logger.debug("Image upscaled: %dx%d → %dx%d", w, h, new_w, min_height)
    return image


def extract_text_from_image(file_path: str) -> str:
    """
    Extract text from an image file using OCR.

    Args:
        file_path: Absolute path to the image file.

    Returns:
        Extracted text as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded as an image.
        RuntimeError: If OCR extraction fails.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    logger.info("Starting OCR on image: %s", path.name)

    # Load image
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {file_path}")

    # Scale up small images
    image = _scale_image_if_small(image)

    # Preprocess
    processed = _preprocess_image(image)

    # OCR configuration — OEM 3 (LSTM + legacy), PSM 3 (auto page segmentation)
    custom_config = r"--oem 3 --psm 3"

    try:
        text = pytesseract.image_to_string(processed, config=custom_config, lang="eng")
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract is not installed or not in PATH. "
            "Install from https://github.com/UB-Mannheim/tesseract/wiki"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"OCR extraction failed: {exc}") from exc

    extracted = text.strip()
    logger.info("OCR complete. Extracted %d characters.", len(extracted))
    return extracted
