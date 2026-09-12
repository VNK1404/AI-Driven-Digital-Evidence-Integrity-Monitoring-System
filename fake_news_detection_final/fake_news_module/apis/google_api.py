"""
apis/google_api.py — Google Fact Check Tools API integration
AI-Driven Digital Evidence Integrity Monitoring System
"""

import logging
import time
import requests

from fake_news_module.config import (
    GOOGLE_API_KEY,
    GOOGLE_FACT_CHECK_URL,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
)

logger = logging.getLogger(__name__)

# Mapping of Google rating strings → canonical labels
_REAL_RATINGS = frozenset({
    "true", "mostly true", "correct", "accurate", "verified",
    "confirmed", "fact", "supported",
})
_FAKE_RATINGS = frozenset({
    "false", "mostly false", "incorrect", "inaccurate", "misleading",
    "disputed", "fabricated", "pants on fire", "fake", "debunked",
    "not true", "wrong",
})


def _parse_rating(rating_str: str) -> str:
    """
    Convert a Google Fact Check rating string to a canonical label.

    Returns: 'Real', 'Fake', or 'Misleading' / 'Unknown'
    """
    lower = rating_str.lower().strip()
    if any(r in lower for r in _REAL_RATINGS):
        return "Real"
    if any(r in lower for r in _FAKE_RATINGS):
        if "mislead" in lower:
            return "Misleading"
        return "Fake"
    return "Unknown"


def call_google_fact_check(text: str) -> str:
    """
    Query the Google Fact Check Tools API for claim reviews.

    Args:
        text: Claim text to search for.

    Returns:
        One of: 'Real', 'Fake', 'Misleading', 'Unknown'
    """
    if not text or not text.strip():
        logger.warning("call_google_fact_check: empty text, returning Unknown.")
        return "Unknown"

    # Use only the first 200 chars as the query (API has length limits)
    query = text[:200].strip()

    params = {
        "key": GOOGLE_API_KEY,
        "query": query,
        "languageCode": "en",
        "pageSize": 5,
    }

    last_error: Exception = RuntimeError("No attempts made.")

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            logger.debug("Google Fact Check API attempt %d …", attempt)
            response = requests.get(
                GOOGLE_FACT_CHECK_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()

            claims = data.get("claims", [])
            if not claims:
                logger.info("Google Fact Check: no matching claims found.")
                return "Unknown"

            # Aggregate ratings from all returned claim reviews
            labels = []
            for claim in claims:
                for review in claim.get("claimReview", []):
                    rating = review.get("textualRating", "")
                    if rating:
                        labels.append(_parse_rating(rating))

            if not labels:
                return "Unknown"

            # Majority vote
            result = _majority_label(labels)
            logger.info(
                "Google Fact Check result: '%s' (from %d reviews: %s)",
                result, len(labels), labels,
            )
            return result

        except requests.exceptions.Timeout:
            logger.warning("Google Fact Check API timeout on attempt %d.", attempt)
            last_error = RuntimeError("Google API timed out.")
        except requests.exceptions.HTTPError as exc:
            logger.error("Google Fact Check HTTP error: %s", exc)
            last_error = exc
            if response.status_code == 400:
                break                       # Bad request — don't retry
        except requests.exceptions.RequestException as exc:
            logger.warning("Google Fact Check request error: %s", exc)
            last_error = exc

        if attempt <= MAX_RETRIES:
            time.sleep(1.5 * attempt)

    logger.error("Google Fact Check failed: %s", last_error)
    return "Unknown"


def _majority_label(labels: list) -> str:
    """Return the most frequent label; ties go to 'Unknown'."""
    counts: dict = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    return max(counts, key=counts.get)
