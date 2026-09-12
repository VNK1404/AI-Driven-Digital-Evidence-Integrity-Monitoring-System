"""
core/normalization.py — Convert API labels to numeric scores
AI-Driven Digital Evidence Integrity Monitoring System
"""

import logging

logger = logging.getLogger(__name__)

# Score mapping: Real → +1, Fake → −1, Uncertain/Misleading/Unknown → 0
LABEL_SCORE_MAP: dict = {
    "real":        +1.0,
    "fake":        -1.0,
    "misleading":  -0.5,   # Partially penalized
    "uncertain":    0.0,
    "unknown":      0.0,
}


def normalize_result(result: str) -> float:
    """
    Convert a string label from any API to a numeric score.

    Args:
        result: Label string. Case-insensitive.

    Returns:
        Float score in range [-1.0, +1.0].
        Returns 0.0 for any unrecognized label.
    """
    key = result.lower().strip()
    score = LABEL_SCORE_MAP.get(key, 0.0)
    logger.debug("normalize_result: '%s' → %.1f", result, score)
    return score
