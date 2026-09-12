"""
apis/async_client.py
====================
Asynchronous external API client for Phase 5.

This module runs Google Fact Check, NewsAPI, and Groq API concurrently with
aiohttp while preserving the labels and article structures consumed by the
existing pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Tuple

import aiohttp

from fake_news_module.cache import get_cache_manager
from fake_news_module.config import (
    API_TIMEOUT_SECONDS,
    GOOGLE_API_KEY,
    GOOGLE_FACT_CHECK_URL,
    GROQ_API_KEY,
    GROQ_API_URL,
    GROQ_MODEL,
    MAX_RETRIES,
    NEWS_API_KEY,
    NEWS_API_URL,
    TRUSTED_SOURCES,
    TRUSTED_SOURCE_THRESHOLD,
)

logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]

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
    lower = rating_str.lower().strip()
    if any(rating in lower for rating in _REAL_RATINGS):
        return "Real"
    if any(rating in lower for rating in _FAKE_RATINGS):
        if "mislead" in lower:
            return "Misleading"
        return "Fake"
    return "Unknown"


def _majority_label(labels: List[str]) -> str:
    counts: Dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return max(counts, key=counts.get)


def _extract_keywords(text: str, max_words: int = 6) -> str:
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "in", "on", "at",
        "to", "for", "of", "and", "or", "but", "that", "this", "with",
        "it", "by", "from", "as", "be", "has", "have", "had", "not",
        "no", "do", "does", "did", "its",
    }
    words = [
        word for word in text.lower().split()
        if word.isalpha() and word not in stopwords and len(word) > 3
    ]
    return " ".join(words[:8])


def _analyze_news_results(articles: List[JsonDict]) -> str:
    if not articles:
        logger.info("async NewsAPI analysis: no articles -> Uncertain")
        return "Uncertain"

    trusted_count = 0
    for article in articles:
        source_name = (
            article.get("source", {}).get("name", "")
            or article.get("source", {}).get("id", "")
        ).lower()

        if any(trusted in source_name for trusted in TRUSTED_SOURCES):
            trusted_count += 1

    if trusted_count >= TRUSTED_SOURCE_THRESHOLD:
        return "Real"
    return "Uncertain"


async def _fetch_json(
    session: aiohttp.ClientSession,
    api_name: str,
    url: str,
    params: JsonDict,
) -> JsonDict | None:
    """Fetch JSON with timeout, retries, exponential backoff, and logging."""
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        start = time.perf_counter()
        try:
            logger.info("%s API start (attempt %d/%d)", api_name, attempt, MAX_RETRIES)
            async with session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                elapsed = time.perf_counter() - start
                logger.info("%s API finish in %.2fs", api_name, elapsed)
                return data

        except asyncio.TimeoutError as exc:
            elapsed = time.perf_counter() - start
            logger.warning("%s API timeout after %.2fs on attempt %d", api_name, elapsed, attempt)
            last_error = exc
        except aiohttp.ClientResponseError as exc:
            elapsed = time.perf_counter() - start
            logger.warning(
                "%s API HTTP failure status=%s after %.2fs on attempt %d",
                api_name,
                exc.status,
                elapsed,
                attempt,
            )
            last_error = exc
            if 400 <= exc.status < 500:
                break
        except aiohttp.ClientError as exc:
            elapsed = time.perf_counter() - start
            logger.warning("%s API client failure after %.2fs on attempt %d: %s", api_name, elapsed, attempt, exc)
            last_error = exc
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - start
            logger.exception("%s API unexpected failure after %.2fs on attempt %d", api_name, elapsed, attempt)
            last_error = exc

        if attempt < MAX_RETRIES:
            delay = 2 ** (attempt - 1)
            logger.info("%s API retry in %ss", api_name, delay)
            await asyncio.sleep(delay)

    logger.error("%s API failed after %d attempt(s): %s", api_name, MAX_RETRIES, last_error)
    return None


async def async_google_factcheck(session: aiohttp.ClientSession, claim: str) -> str:
    """Return Google Fact Check label: Real, Fake, Misleading, or Unknown."""
    if not claim or not claim.strip():
        logger.warning("async_google_factcheck: empty claim")
        return "Unknown"

    params = {
        "key": GOOGLE_API_KEY,
        "query": claim[:200].strip(),
        "languageCode": "en",
        "pageSize": 5,
    }
    data = await _fetch_json(session, "Google Fact Check", GOOGLE_FACT_CHECK_URL, params)
    if not data:
        return "Unknown"

    labels: List[str] = []
    for fact_claim in data.get("claims", []):
        for review in fact_claim.get("claimReview", []):
            rating = review.get("textualRating", "")
            if rating:
                labels.append(_parse_rating(rating))

    if not labels:
        logger.info("Google Fact Check: no usable claim ratings found")
        return "Unknown"

    result = _majority_label(labels)
    logger.info("Google Fact Check async result: '%s' from %d review(s)", result, len(labels))
    return result


async def async_newsapi(session: aiohttp.ClientSession, claim: str) -> Tuple[str, List[JsonDict]]:
    """Return NewsAPI tuple: (label, articles)."""
    if not claim or not claim.strip():
        logger.warning("async_newsapi: empty claim")
        return ("Unknown", [])

    query = _extract_keywords(claim)
    if not query:
        logger.warning("async_newsapi: no keywords extracted")
        return ("Unknown", [])

    params = {
        "q": query,
        "apiKey": NEWS_API_KEY,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 20,
    }
    data = await _fetch_json(session, "NewsAPI", NEWS_API_URL, params)
    if not data or data.get("status") != "ok":
        return ("Unknown", [])

    articles = data.get("articles", [])
    if not isinstance(articles, list):
        return ("Unknown", [])

    label = _analyze_news_results(articles)
    logger.info("NewsAPI async result: '%s' from %d article(s)", label, len(articles))
    return (label, articles)


async def async_groq(session: aiohttp.ClientSession, claim: str) -> str:
    """Query Groq API to evaluate a claim. Returns: Real, Fake, Uncertain, or Unknown."""
    if not claim or not claim.strip():
        logger.warning("async_groq: empty claim")
        return "Unknown"
    if not GROQ_API_KEY:
        logger.error("Groq API key not configured")
        return "Unknown"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GROQ_API_KEY}",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a fact-checking assistant. "
                    "When given a claim, respond with exactly one word: "
                    "'Real', 'Fake', or 'Uncertain'. No explanation."
                ),
            },
            {
                "role": "user",
                "content": f"Is this claim real or fake? Claim: {claim}",
            },
        ],
        "temperature": 0,
    }
    try:
        async with session.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            data = await response.json()
            text = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
                .lower()
            )
            logger.info("Groq async response: '%s'", text)
            if "real" in text or "true" in text:
                return "Real"
            if "fake" in text or "false" in text:
                return "Fake"
            if "uncertain" in text:
                return "Uncertain"
            return "Unknown"
    except Exception as exc:  # noqa: BLE001
        logger.error("Groq async exception: %s", exc)
        return "Unknown"


async def run_external_apis_async(claim: str) -> Dict[str, Any]:
    """Run all external APIs concurrently and return pipeline-compatible keys."""
    cache = get_cache_manager()
    cached = cache.get_api_response("external_apis", claim)
    if isinstance(cached, dict):
        logger.info("External API cache hit")
        return {
            "google": _coerce_label("google", cached.get("google"), "Unknown"),
            "news": _coerce_news_result(cached.get("news")),
            "groq": _coerce_label("groq", cached.get("groq"), "Unknown"),
        }

    timeout = aiohttp.ClientTimeout(total=API_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        google_result, news_result, groq_result = await asyncio.gather(
            async_google_factcheck(session, claim),
            async_newsapi(session, claim),
            async_groq(session, claim),
            return_exceptions=True,
        )

    result = {
        "google": _coerce_label("google", google_result, "Unknown"),
        "news": _coerce_news_result(news_result),
        "groq": _coerce_label("groq", groq_result, "Unknown"),
    }
    cache.set_api_response("external_apis", claim, result)
    return result


def _coerce_label(name: str, result: Any, fallback: str) -> str:
    if isinstance(result, Exception):
        logger.error("%s async task failed: %s", name, result)
        return fallback
    if isinstance(result, str):
        return result
    logger.error("%s async task returned unexpected type: %s", name, type(result).__name__)
    return fallback


def _coerce_news_result(result: Any) -> Tuple[str, List[JsonDict]]:
    if isinstance(result, Exception):
        logger.error("news async task failed: %s", result)
        return ("Unknown", [])
    if (
        isinstance(result, tuple)
        and len(result) == 2
        and isinstance(result[0], str)
        and isinstance(result[1], list)
    ):
        return result
    if (
        isinstance(result, list)
        and len(result) == 2
        and isinstance(result[0], str)
        and isinstance(result[1], list)
    ):
        return (result[0], result[1])
    logger.error("news async task returned unexpected type: %s", type(result).__name__)
    return ("Unknown", [])
