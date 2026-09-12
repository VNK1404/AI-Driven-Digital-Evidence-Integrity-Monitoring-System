"""
fake_news_module/similarity/inference.py
==========================================
Thin bridge between the pipeline and the FAISS similarity searcher.

This is the ONLY import the pipeline needs from the similarity subpackage.
It returns the same simple string labels ("Real", "Fake", "Uncertain") that
all other API wrappers return, making it a drop-in addition to _API_TASKS.

Usage (internal)
----------------
    from fake_news_module.similarity.inference import call_similarity

    label = call_similarity("Senate panel to hold hearing on Equifax hack")
    # → "Real"
"""

import logging
from fake_news_module.similarity.searcher import search

logger = logging.getLogger(__name__)


def call_similarity(claim: str) -> str:
    """
    Run FAISS similarity search on a claim and return a standardized label.

    This function is intentionally simple — all complexity lives in
    searcher.search(). The pipeline calls this exactly like any other
    API wrapper.

    Args:
        claim: The extracted news claim text.

    Returns:
        One of:
            "Real"      — majority of top-K similar claims are real news
            "Fake"      — majority of top-K similar claims are fake news
            "Uncertain" — index not built / tie vote / low similarity scores
    """
    if not claim or not claim.strip():
        logger.warning("call_similarity: empty claim — returning Uncertain.")
        return "Uncertain"

    result = search(claim)
    label      = result.get("label", "Uncertain")
    confidence = result.get("confidence", 0.0)
    n_matches  = len(result.get("top_matches", []))

    logger.info(
        "call_similarity: label='%s' confidence=%.4f top_matches=%d",
        label, confidence, n_matches,
    )
    return label
