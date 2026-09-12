"""
fake_news_module/explainability/confidence_breakdown.py
=========================================================
Generates a percentage-based breakdown of each signal's contribution
to the final weighted decision score.

The breakdown shows how much each verification channel influenced
the final verdict, which helps users understand which signals drove
the classification.

Usage
-----
    from fake_news_module.explainability.confidence_breakdown import build_confidence_breakdown

    breakdown = build_confidence_breakdown(
        api_results={
            "roberta":    "Fake",
            "similarity": "Fake",
            "google":     "Unknown",
            "news":       "Uncertain",
            "groq":       "Unknown",
            "heuristic":  "Uncertain",
        },
        source_score=0.75,
    )
    # → {
    #     "roberta":    {"weight_pct": 30, "label": "Fake",     "contributed": True},
    #     "similarity": {"weight_pct": 22, "label": "Fake",     "contributed": True},
    #     "google":     {"weight_pct": 18, "label": "Unknown",  "contributed": False},
    #     "news":       {"weight_pct": 10, "label": "Uncertain","contributed": False},
    #     "groq":       {"weight_pct": 25, "label": "Unknown",  "contributed": False},
    #     "heuristic":  {"weight_pct": 5,  "label": "Uncertain","contributed": False},
    #     "source":     {"weight_pct": 10, "label": "Moderate", "contributed": True},
    # }
"""

import logging
from typing import Any, Dict, Optional

from fake_news_module.config import API_WEIGHTS, SOURCE_WEIGHT

logger = logging.getLogger(__name__)

# Labels that carry no meaningful signal and should be marked as non-contributing
_NON_CONTRIBUTING_LABELS = {"unknown", "uncertain", ""}


def build_confidence_breakdown(
    api_results: Dict[str, str],
    source_score: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Build a human-readable confidence breakdown for each verification signal.

    Each signal entry contains:
        - weight_pct   : int  — percentage weight this signal carries in the model
        - label        : str  — the raw verdict returned by this signal
        - contributed  : bool — whether the signal returned a meaningful result
        - direction    : str  — "REAL" | "FAKE" | "NEUTRAL" — direction of evidence

    Args:
        api_results:  Dict mapping signal names to label strings.
                      Expected keys: roberta, similarity, google, news, groq, heuristic
        source_score: Optional float [0, 1] — aggregate source credibility score.
                      When provided, it is included as the "source" signal.

    Returns:
        Dict[signal_name → breakdown_dict]
    """
    # Combined weights from config — sum should equal 1.0 (minus source weight)
    raw_weights: Dict[str, float] = {
        "roberta":    API_WEIGHTS.get("roberta",    0.20),
        "similarity": API_WEIGHTS.get("similarity", 0.20),
        "google":     API_WEIGHTS.get("google",     0.20),
        "news":       API_WEIGHTS.get("news",       0.10),
        "groq":       API_WEIGHTS.get("groq",       0.25),
        "heuristic":  API_WEIGHTS.get("heuristic",  0.05),
    }

    # Add source credibility weight if provided
    if source_score is not None:
        raw_weights["source"] = SOURCE_WEIGHT

    # Normalise to percentages
    total = sum(raw_weights.values())
    if total == 0:
        logger.warning("confidence_breakdown: total weight is 0, using equal weights.")
        total = 1.0

    breakdown: Dict[str, Dict[str, Any]] = {}

    for signal, weight in raw_weights.items():
        weight_pct = round((weight / total) * 100)

        if signal == "source":
            label = _source_label(source_score)
            contributed = source_score is not None and source_score > 0.0
            direction = _source_direction(source_score)
        else:
            label = api_results.get(signal, "Unknown")
            contributed = label.lower().strip() not in _NON_CONTRIBUTING_LABELS
            direction = _label_direction(label)

        breakdown[signal] = {
            "weight_pct":  weight_pct,
            "label":       label,
            "contributed": contributed,
            "direction":   direction,
        }

        logger.debug(
            "confidence_breakdown: signal='%s' weight_pct=%d label='%s' "
            "contributed=%s direction='%s'",
            signal, weight_pct, label, contributed, direction,
        )

    logger.info(
        "confidence_breakdown: built breakdown for %d signals.", len(breakdown)
    )
    return breakdown


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _label_direction(label: str) -> str:
    """Map a raw API label to a directional tag: REAL, FAKE, or NEUTRAL."""
    key = label.lower().strip()
    if key in {"real", "true"}:
        return "REAL"
    if key in {"fake", "false", "misleading"}:
        return "FAKE"
    return "NEUTRAL"


def _source_label(score: Optional[float]) -> str:
    """Convert a numeric source credibility score to a human-readable label."""
    if score is None:
        return "Unknown"
    if score >= 0.90:
        return "Very High"
    if score >= 0.75:
        return "High"
    if score >= 0.55:
        return "Moderate"
    if score >= 0.35:
        return "Low"
    return "Very Low"


def _source_direction(score: Optional[float]) -> str:
    """Treat high source credibility as evidence towards REAL."""
    if score is None:
        return "NEUTRAL"
    if score >= 0.70:
        return "REAL"
    if score <= 0.40:
        return "FAKE"
    return "NEUTRAL"
