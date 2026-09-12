"""
ocr/extractor.py — Unified file-type aware text extractor
AI-Driven Digital Evidence Integrity Monitoring System
"""

import logging
from pathlib import Path

from fake_news_module.config import (
    SUPPORTED_IMAGE_EXTENSIONS,
    SUPPORTED_PDF_EXTENSIONS,
)
from fake_news_module.ocr.image_ocr import extract_text_from_image
from fake_news_module.ocr.pdf_reader import extract_text_from_pdf

logger = logging.getLogger(__name__)

# Union of all accepted extensions
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS


def extract_text(file_path: str) -> str:
    """
    Unified entry point: detect file type and delegate to the correct extractor.

    Args:
        file_path: Absolute path to an image or PDF file.

    Returns:
        Raw extracted text string.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the file extension is not supported.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{suffix}'. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    logger.info("Detected file type '%s' for: %s", suffix, path.name)

    if suffix in SUPPORTED_IMAGE_EXTENSIONS:
        return extract_text_from_image(file_path)

    if suffix in SUPPORTED_PDF_EXTENSIONS:
        return extract_text_from_pdf(file_path)

    # Unreachable — kept for safety
    raise ValueError(f"Unhandled extension: {suffix}")
