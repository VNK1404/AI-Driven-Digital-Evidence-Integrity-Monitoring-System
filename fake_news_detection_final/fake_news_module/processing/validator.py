"""
processing/validator.py — OCR quality validation
AI-Driven Digital Evidence Integrity Monitoring System
"""

import re
import logging
from typing import Tuple

from fake_news_module.config import MIN_TEXT_LENGTH, MAX_NOISE_RATIO

logger = logging.getLogger(__name__)


def _compute_noise_ratio(text: str) -> float:
    """
    Noise ratio = (non-alphanumeric non-space characters) / total characters.
    A high ratio suggests garbled OCR output.
    """
    if not text:
        return 1.0
    noise_chars = len(re.findall(r"[^a-z0-9\s]", text))
    return noise_chars / len(text)


def validate_text(text: str) -> Tuple[bool, str]:
    """
    Validate OCR output quality.

    Rules:
        1. Length must exceed MIN_TEXT_LENGTH (default: 30 characters).
        2. Noise ratio must be below MAX_NOISE_RATIO (default: 0.4).

    Args:
        text: Cleaned text string.

    Returns:
        Tuple of (is_valid: bool, reason: str).
        reason is empty string when valid.

    Raises:
        TypeError: If text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"validate_text expects str, got {type(text).__name__}")

    # Rule 1 — minimum length
    if len(text) < MIN_TEXT_LENGTH:
        reason = (
            f"Text too short: {len(text)} characters "
            f"(minimum required: {MIN_TEXT_LENGTH})."
        )
        logger.warning("Validation failed: %s", reason)
        return False, reason

    # Rule 2 — noise ratio
    noise = _compute_noise_ratio(text)
    if noise >= MAX_NOISE_RATIO:
        reason = (
            f"Noise ratio too high: {noise:.2f} "
            f"(maximum allowed: {MAX_NOISE_RATIO}). "
            "Possible garbled OCR output."
        )
        logger.warning("Validation failed: %s", reason)
        return False, reason

    logger.debug(
        "Text validated: length=%d, noise_ratio=%.2f", len(text), noise
    )
    return True, ""
