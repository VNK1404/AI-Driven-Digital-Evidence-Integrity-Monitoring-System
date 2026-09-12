"""
processing/claim_extractor.py — Extract main claim/headline from cleaned text
AI-Driven Digital Evidence Integrity Monitoring System
"""

import re
import logging
from typing import Optional

from fake_news_module.config import CLAIM_MIN_LENGTH, CLAIM_FALLBACK_LENGTH

logger = logging.getLogger(__name__)

# Sentence boundary pattern
_SENTENCE_SPLITTER = re.compile(r"(?<=[.!?])\s+|\n+")

# Patterns to SKIP — social media headers, usernames, noise
_SKIP_PATTERNS = [
    re.compile(r"@\w+"),                              # Twitter handles
    re.compile(r"^\s*[\w\s]{1,30}\s*@"),              # "Name @handle" lines
    re.compile(r"^[\W\d\s]{0,5}$"),                   # Pure symbols/digits
    re.compile(r"verified|retweet|like|follow|share", re.I),
    re.compile(r"^\s*(rt|via|cc)\s", re.I),           # RT / via
    re.compile(r"^[\u0000-\u001F\u2000-\u2FFF\s]+$"), # Emoji/control-only lines
    re.compile(r"^\s*[#\-_=*~`]+\s*$"),               # Separator lines
    re.compile(r"^[a-z]{1,3}\s+[a-z]{1,10}\s+[a-z]{1,5}\s+[a-z]{1,5}",), # short janky OCR header
]

# News-signal words that indicate a real claim/headline
_NEWS_SIGNALS = re.compile(
    r"\b(confirmed|official|report|breaking|war|attack|dies|killed|"
    r"arrested|closed|signed|approved|launched|failed|won|lost|"
    r"president|minister|government|military|ceasefire|deal|"
    r"billion|million|percent|crisis|ban|sanction|election|"
    r"strait|nuclear|missile|troops|forces|police|court|passes)\b",
    re.I,
)


def _is_noise_line(sentence: str) -> bool:
    """Return True if the sentence looks like a noisy header or social media artifact."""
    for pat in _SKIP_PATTERNS:
        if pat.search(sentence):
            return True
    return False


def _score_sentence(sentence: str) -> float:
    """
    Score a sentence for how likely it is to be the main claim.
    Higher = better candidate.
    """
    score = 0.0
    words = sentence.split()

    # Reward length (up to a cap)
    score += min(len(sentence) / 200, 1.0) * 2

    # Reward word count
    score += min(len(words) / 15, 1.0) * 1.5

    # Reward news signal words
    news_hits = len(_NEWS_SIGNALS.findall(sentence))
    score += news_hits * 2.0

    # Penalize if starts with a lowercase word (likely continuation or username)
    if words and words[0][0].islower():
        score -= 1.0

    # Reward if starts with capitalised word
    if words and words[0][0].isupper():
        score += 0.5

    # Penalize if it contains @ (username artefact)
    if "@" in sentence:
        score -= 5.0

    # Penalize OCR garbage characters
    garbage_ratio = len(re.findall(r"[^a-zA-Z0-9\s.,!?'\"-:$%]", sentence)) / max(len(sentence), 1)
    score -= garbage_ratio * 3.0

    return score


def extract_main_text(text: str) -> str:
    """
    Extract the most likely main claim or headline from text.

    Strategy:
    1. Split into sentences / lines.
    2. Filter out noise lines (social media headers, handles, etc.)
    3. Score remaining sentences by news relevance.
    4. Return the highest-scoring candidate.
    5. Fallback: first CLAIM_FALLBACK_LENGTH characters of non-noise text.

    Args:
        text: Cleaned, validated text.

    Returns:
        Extracted claim string.

    Raises:
        ValueError: If text is empty.
    """
    if not text or not text.strip():
        raise ValueError("Cannot extract claim from empty text.")

    # Split on sentence boundaries AND newlines
    raw_parts = _SENTENCE_SPLITTER.split(text)
    sentences = [s.strip() for s in raw_parts if s.strip()]

    logger.debug("Claim extraction: %d raw parts.", len(sentences))

    # Filter out noise and too-short lines
    candidates = [
        s for s in sentences
        if len(s) >= CLAIM_MIN_LENGTH
        and len(s.split()) >= 4
        and re.search(r"[a-zA-Z]{3,}", s)
        and not _is_noise_line(s)
    ]

    logger.debug("Claim extraction: %d candidates after filtering.", len(candidates))

    if candidates:
        # Pick the highest-scoring sentence
        best = max(candidates, key=_score_sentence)
        logger.info("Claim selected (score=%.2f): '%s...'", _score_sentence(best), best[:100])
        return best

    # Fallback 1: use any sentence that has enough alpha content
    for s in sentences:
        if len(s) >= CLAIM_MIN_LENGTH and re.search(r"[a-zA-Z]{3,}", s):
            logger.warning("Fallback claim (no candidates): '%s...'", s[:80])
            return s

    # Fallback 2: first N characters
    fallback = text[:CLAIM_FALLBACK_LENGTH].strip()
    logger.warning("Using character-slice fallback: '%s...'", fallback[:80])
    return fallback
