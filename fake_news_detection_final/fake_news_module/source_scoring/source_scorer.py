"""
source_scoring/source_scorer.py
=================================
Credibility scoring for news domains found in claims and API evidence.

Plain claim text without URLs is valid input and always returns a neutral
score. The module should never crash during normal pipeline processing.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_CURRENT_DIR = Path(__file__).resolve().parent
_JSON_PATH = _CURRENT_DIR / "source_scores.json"
_CACHE: Optional[Dict[str, float]] = None


def _load_scores() -> Dict[str, float]:
    """Load the JSON mapping domain -> credibility score."""
    if not _JSON_PATH.is_file():
        logger.warning("Source scores file not found at %s; using empty mapping", _JSON_PATH)
        return {}

    try:
        with open(_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(k).lower(): float(v) for k, v in data.items()}
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load source scores from %s: %s", _JSON_PATH, exc)
        return {}


def _get_cache() -> Dict[str, float]:
    global _CACHE
    if _CACHE is None:
        _CACHE = _load_scores()
    return _CACHE


def _extract_domain(text: str) -> str:
    """Extract the first URL hostname from text, or return an empty string."""
    domains = _extract_domains(text)
    return domains[0] if domains else ""


def _extract_domains(text: str) -> List[str]:
    """Extract all URL hostnames from text, preserving first-seen order."""
    if not text:
        return []

    try:
        import re

        domains: List[str] = []
        seen = set()
        for raw_url in re.findall(r"https?://[^\s)>\]\"']+", text):
            parsed = urlparse(raw_url.rstrip(".,;:!?"))
            hostname = (parsed.hostname or "").lower().removeprefix("www.")
            if hostname and hostname not in seen:
                seen.add(hostname)
                domains.append(hostname)
        return domains
    except Exception as exc:  # noqa: BLE001
        logger.debug("Domain extraction error for text %s: %s", text, exc)
        return []


def _score_domain(domain: str, scores: Dict[str, float]) -> float:
    raw = scores.get(domain, scores.get("unknown", 0.5))
    return max(0.0, min(1.0, float(raw)))


def get_source_score(text: str) -> float:
    """Return a credibility score for the first URL in text.

    Unknown domains and ordinary claim text without URLs return 0.5. This
    compatibility function is intentionally non-throwing for pipeline safety.
    """
    domain = _extract_domain(text)
    if not domain:
        logger.debug("No domain could be extracted; returning neutral source score")
        return 0.5

    scores = _get_cache()
    score = _score_domain(domain, scores)
    logger.info("Source credibility for domain '%s' resolved to %.2f", domain, score)
    return score


def score_source_evidence(items: Iterable[Any]) -> Dict[str, Any]:
    """Score all URLs found in claim/API evidence.

    Args:
        items: Strings or article-like dictionaries that may contain URLs.

    Returns:
        A JSON-ready dict with a numeric ``source_score`` and ``sources_found``.
    """
    scores = _get_cache()
    sources_found: List[Dict[str, Any]] = []
    seen = set()

    for item in items:
        texts: List[str] = []
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            for key in ("url", "link", "webUrl", "source_url"):
                value = item.get(key)
                if isinstance(value, str):
                    texts.append(value)

            source = item.get("source")
            if isinstance(source, dict):
                value = source.get("url")
                if isinstance(value, str):
                    texts.append(value)

        for text in texts:
            for domain in _extract_domains(text):
                if domain in seen:
                    continue
                seen.add(domain)
                sources_found.append({
                    "domain": domain,
                    "score": round(_score_domain(domain, scores), 4),
                })

    if not sources_found:
        return {"source_score": 0.5, "sources_found": []}

    average = sum(source["score"] for source in sources_found) / len(sources_found)
    return {
        "source_score": round(average, 4),
        "sources_found": sources_found,
    }
