"""
apis/news_api.py — NewsAPI.org integration + trusted-source analysis
AI-Driven Digital Evidence Integrity Monitoring System
"""

import logging
import time
from typing import List, Dict, Any

import requests

from fake_news_module.config import (
    NEWS_API_KEY,
    NEWS_API_URL,
    TRUSTED_SOURCES,
    TRUSTED_SOURCE_THRESHOLD,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
)

logger = logging.getLogger(__name__)


def _extract_keywords(text: str, max_words: int = 6) -> str:
    """
    Extract the most significant words from a claim for use as a search query.
    Strips common stopwords and limits the query length.
    """
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "in", "on", "at",
        "to", "for", "of", "and", "or", "but", "that", "this", "with",
        "it", "by", "from", "as", "be", "has", "have", "had", "not",
        "no", "do", "does", "did", "its",
    }
    words = [
        w for w in text.lower().split()
        if w.isalpha() and w not in stopwords and len(w) > 3
    ]
    # Take up to max_words most meaningful words
    return " ".join(words[:8])


def call_news_api(text: str) -> List[Dict[str, Any]]:
    """
    Search NewsAPI.org for articles related to the claim.

    Args:
        text: Claim text to search for.

    Returns:
        List of article dicts (may be empty).
        Each dict contains at minimum: source, title, url.
    """
    if not text or not text.strip():
        logger.warning("call_news_api: empty text, returning [].")
        return []

    query = _extract_keywords(text)
    if not query:
        logger.warning("call_news_api: could not extract keywords.")
        return []

    params = {
        "q": query,
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 20,
    }

    last_error: Exception = RuntimeError("No attempts made.")

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            logger.debug(
                "NewsAPI attempt %d, query='%s' …", attempt, query
            )
            response = requests.get(
                NEWS_API_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                logger.error(
                    "NewsAPI non-ok status: %s", data.get("message", "unknown")
                )
                return []

            articles = data.get("articles", [])
            logger.info(
                "NewsAPI returned %d articles for query: '%s'",
                len(articles), query,
            )
            return articles

        except requests.exceptions.Timeout:
            logger.warning("NewsAPI timeout on attempt %d.", attempt)
            last_error = RuntimeError("NewsAPI timed out.")
        except requests.exceptions.HTTPError as exc:
            logger.error("NewsAPI HTTP error: %s", exc)
            last_error = exc
            break
        except requests.exceptions.RequestException as exc:
            logger.warning("NewsAPI request error: %s", exc)
            last_error = exc

        if attempt <= MAX_RETRIES:
            time.sleep(1.5 * attempt)

    logger.error("NewsAPI failed: %s", last_error)
    return []


def analyze_news_results(articles: List[Dict[str, Any]]) -> str:
    """
    Analyze NewsAPI results based on trusted source coverage.

    Logic:
    - Count articles whose source name contains a trusted source keyword.
    - If count >= TRUSTED_SOURCE_THRESHOLD → Real
    - Else → Uncertain

    Args:
        articles: List of article dicts from call_news_api().

    Returns:
        'Real' or 'Uncertain'
    """
    if not articles:
        logger.info("analyze_news_results: no articles to analyze → Uncertain.")
        return "Uncertain"

    trusted_count = 0
    trusted_found: List[str] = []

    for article in articles:
        source_name = (
            article.get("source", {}).get("name", "")
            or article.get("source", {}).get("id", "")
        ).lower()

        for trusted in TRUSTED_SOURCES:
            if trusted in source_name:
                trusted_count += 1
                trusted_found.append(source_name)
                break                   # count each article once

    logger.info(
        "Trusted sources found: %d / %d (threshold=%d). Sources: %s",
        trusted_count,
        len(articles),
        TRUSTED_SOURCE_THRESHOLD,
        trusted_found,
    )

    if trusted_count >= TRUSTED_SOURCE_THRESHOLD:
        return "Real"
    return "Uncertain"
