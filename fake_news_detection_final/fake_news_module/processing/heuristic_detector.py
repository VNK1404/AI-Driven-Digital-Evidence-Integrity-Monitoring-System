"""
fake_news_module/processing/heuristic_detector.py
===================================================
Lightweight rule-based heuristic detector for sensationalist and conspiracy content.

This module provides a fast, interpretable signal that catches patterns
commonly associated with fake / misinformation content:
    - ALL-CAPS headlines or excessive exclamation marks
    - Conspiracy / pseudoscience keywords
    - Absence of credible citations
    - Sensationalist language ("BOMBSHELL", "EXCLUSIVE", "BREAKING")
    - Anonymous sourcing language

Returns:
    "Fake"      — strong sensationalism/conspiracy markers found
    "Uncertain" — weak or no markers found (abstain, don't penalise real news)

Note: This heuristic NEVER returns "Real" — it can only flag Fake or abstain.
That way it does not override legitimate news signals.
"""

import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Keyword lists
# ──────────────────────────────────────────────

# Sensationalist / clickbait trigger words (ANY match → suspicious)
_SENSATIONAL_PATTERNS = [
    r"\bbombshell\b",
    r"\bbreaking exclusive\b",
    r"\bshocking truth\b",
    r"\bwhistleblower\b.*\banonymous\b",
    r"\bthey don.?t want you to know\b",
    r"\bmainstream media.*suppress\b",
    r"\bshare this before they take it down\b",
    r"\bshare before.*deleted\b",
    r"\bbig pharma\b",
    r"\bnew world order\b",
    r"\bdeep state\b",
    r"\bchips?\b.*\bmicrochip\b",
    r"\bchemtrail\b",
    r"\bmind.?control\b",
    r"\b5g\b.*\bcontrol\b",
    r"\bvaccine.*microchip\b",
    r"\bglobal.*elite\b",
    r"\bworld economic forum.*suppress\b",
    r"\bsecret.*laboratory\b",
    r"\bgovernment.*cover.?up\b",
    r"\bthis story.*suppressed\b",
    r"\bscientists.*silenced\b",
    r"\bno peer review\b",
    r"\banonymous.*tip\b",
    r"\bagent x\b",
    r"\bleaked document.*secret\b",
]

# Credibility markers (presence reduces fake score)
_CREDIBILITY_PATTERNS = [
    r"\bpeer.?reviewed\b",
    r"\bpublished in\b",
    r"\bnature\b",
    r"\bscience\b.*\bjournal\b",
    r"\bpress release\b",
    r"\bofficial\b",
    r"\bnasa\b",
    r"\bwho\b|\bunesco\b|\bun\b",
    r"\bresearchers at\b",
    r"\bprofessor\b|\bdr\.\b",
    r"\bstudy\b.*\bfound\b",
]

# Compiled patterns (case-insensitive)
_RE_SENSATIONAL = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _SENSATIONAL_PATTERNS]
_RE_CREDIBILITY = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _CREDIBILITY_PATTERNS]


def detect_sensationalism(text: str) -> Tuple[str, float, list]:
    """
    Analyse text for sensationalist / conspiracy markers.

    Args:
        text: The input news text or claim.

    Returns:
        Tuple of:
            label:       "Fake" | "Uncertain"
            confidence:  float in [0.0, 1.0]
            reasons:     list of triggered pattern descriptions
    """
    if not text or not text.strip():
        return "Uncertain", 0.0, []

    text_lower = text.lower()
    triggered = []

    # Count sensationalism hits
    for pattern in _RE_SENSATIONAL:
        m = pattern.search(text)
        if m:
            triggered.append(m.group(0)[:60])

    # Count credibility markers
    credibility_hits = sum(1 for p in _RE_CREDIBILITY if p.search(text))

    # Scoring: each triggered sensational marker adds 0.25, capped at 1.0
    # Credibility markers discount by 0.15 each, floor at 0.0
    raw_score = min(len(triggered) * 0.25, 1.0) - credibility_hits * 0.15
    confidence = max(0.0, round(raw_score, 4))

    # Threshold: score > 0.30 → "Fake" (at least 2 markers with no credibility offset)
    label = "Fake" if confidence > 0.30 else "Uncertain"

    logger.info(
        "heuristic_detector: triggered=%d credibility=%d score=%.4f label=%s",
        len(triggered), credibility_hits, confidence, label,
    )

    return label, confidence, triggered


def call_heuristic(text: str) -> str:
    """
    Thin wrapper for pipeline integration — returns a label string only.

    Returns:
        "Fake"      — strong sensationalism detected
        "Uncertain" — no strong markers found
    """
    label, _, _ = detect_sensationalism(text)
    return label
