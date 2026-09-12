"""
processing/text_cleaner.py -- Text normalization and cleaning
AI-Driven Digital Evidence Integrity Monitoring System
"""

import re
import logging
import unicodedata

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# OCR compound-word restoration map
# PDF renderers and Tesseract often merge hyphenated words into one token.
# This map restores the most common cases so similarity embeddings match
# the training corpus (which contains properly hyphenated text).
# ──────────────────────────────────────────────
_COMPOUND_FIXES = {
    "peerreviewed":     "peer-reviewed",
    "longterm":         "long-term",
    "shortterm":        "short-term",
    "highquality":      "high-quality",
    "lowquality":       "low-quality",
    "realtime":         "real-time",
    "alltime":          "all-time",
    "wellknown":        "well-known",
    "wellbeing":        "well-being",
    "mindcontrol":      "mind-control",
    "mindcontrolling":  "mind-controlling",
    "factcheck":        "fact-check",
    "factchecking":     "fact-checking",
    "followup":         "follow-up",
    "socialmedia":      "social media",
    "selfdriving":      "self-driving",
    "largescale":       "large-scale",
    "smallscale":       "small-scale",
    "opensource":       "open-source",
    "worldclass":       "world-class",
    "datadriven":       "data-driven",
    "evidencebased":    "evidence-based",
    "awardwinning":     "award-winning",
    "groundbreaking":   "ground-breaking",
    "lifechanging":     "life-changing",
}

# Build a single regex that matches any of the keys (word-boundary anchored)
_COMPOUND_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _COMPOUND_FIXES) + r")\b",
    re.IGNORECASE,
)


def _fix_ocr_hyphenation(text: str) -> str:
    """Restore compound words merged by OCR/PDF rendering."""
    def _replace(m: re.Match) -> str:
        token = m.group(0).lower()
        return _COMPOUND_FIXES.get(token, m.group(0))
    return _COMPOUND_RE.sub(_replace, text)


def _normalize_unicode(text: str) -> str:
    """Normalize unicode characters to ASCII-compatible form."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _remove_special_characters(text: str) -> str:
    """
    Remove non-alphanumeric characters except sentence punctuation.
    Keeps: letters, digits, whitespace, . , ! ? ' " - :
    """
    return re.sub(r"[^a-z0-9\s.,!?'\"\-:]", " ", text)


def _collapse_whitespace(text: str) -> str:
    """Collapse multiple whitespace into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def _remove_ocr_artifacts(text: str) -> str:
    """
    Remove common OCR artifacts:
    - Lone single characters (except 'a', 'i')
    - Repeated punctuation
    - Lines that are pure garbage (all non-alpha)
    """
    # Remove lone characters that aren't common words
    text = re.sub(r"\b(?![aAiI])[^aAiI\s]\b", "", text)
    # Remove lines consisting entirely of non-letter characters
    lines = [
        line for line in text.split("\n")
        if re.search(r"[a-z]{2,}", line)  # at least one 2-letter sequence
    ]
    return "\n".join(lines)


def clean_text(text: str) -> str:
    """
    Full text cleaning pipeline:
    1. Unicode normalization
    2. Lowercase
    3. OCR compound-word restoration  (fixes 'peerreviewed' -> 'peer-reviewed')
    4. OCR artifact removal
    5. Special character removal
    6. Whitespace collapse

    Args:
        text: Raw extracted text.

    Returns:
        Cleaned, normalized text string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")

    logger.debug("clean_text: input length=%d", len(text))

    text = _normalize_unicode(text)
    text = text.lower()
    text = _fix_ocr_hyphenation(text)   # restore compound words before artifact removal
    text = _remove_ocr_artifacts(text)
    text = _remove_special_characters(text)
    text = _collapse_whitespace(text)

    logger.debug("clean_text: output length=%d", len(text))
    return text
