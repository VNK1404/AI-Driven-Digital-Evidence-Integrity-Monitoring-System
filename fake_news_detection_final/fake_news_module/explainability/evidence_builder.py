"""
fake_news_module/explainability/evidence_builder.py
=====================================================
Collects and structures evidence from every verification channel in
the pipeline and presents it in a form suitable for the UI and logs.

The evidence dict is fully JSON-serializable and contains:
    - roberta_evidence:     ML model prediction details
    - similarity_evidence:  Top-K FAISS nearest-neighbour matches
    - factcheck_evidence:   Google Fact-Check label and note
    - newsapi_evidence:     NewsAPI article count and sources
    - groq_evidence:        Groq LLM label and note
    - source_evidence:      Source credibility scores and domains

Usage
-----
    from fake_news_module.explainability.evidence_builder import build_evidence

    evidence = build_evidence(
        claim="Vaccines cause autism",
        api_results={...},
        similarity_top_matches=[...],
        news_articles=[...],
        source_credibility={...},
    )
"""

import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def build_evidence(
    claim: str,
    api_results: Dict[str, str],
    similarity_top_matches: Optional[List[Dict[str, Any]]] = None,
    news_articles: Optional[List[Dict[str, Any]]] = None,
    source_credibility: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assemble structured evidence from all pipeline signals.

    Args:
        claim:                 The extracted news claim text.
        api_results:           Dict of {signal_name: label_string} from pipeline.
        similarity_top_matches: List of top-K FAISS matches
                                [{"title": str, "label": str, "score": float}]
        news_articles:         Raw article dicts from NewsAPI.
        source_credibility:    Output from source_scorer (source_score, sources_found).

    Returns:
        Fully JSON-serializable evidence dict.
    """
    evidence: Dict[str, Any] = {}

    # ── RoBERTa evidence ─────────────────────────────────────────────────────
    roberta_label = api_results.get("roberta", "Unknown")
    evidence["roberta_evidence"] = {
        "label":       roberta_label,
        "model":       "RoBERTa (fine-tuned on WELFake dataset)",
        "description": (
            "Local transformer-based classifier trained on 44,898 news articles. "
            "Evaluates linguistic patterns, phrasing, and semantic content."
        ),
    }

    # ── Similarity evidence ───────────────────────────────────────────────────
    sim_label   = api_results.get("similarity", "Uncertain")
    top_matches = similarity_top_matches or []
    evidence["similarity_evidence"] = {
        "label":        sim_label,
        "top_matches":  top_matches[:5],   # cap at 5 for display
        "match_count":  len(top_matches),
        "description":  (
            "FAISS semantic similarity search over 44,898 labelled news claims "
            "using all-MiniLM-L6-v2 dense embeddings (384-dim cosine similarity)."
        ),
    }
    logger.debug(
        "evidence_builder: similarity=%s top_matches=%d",
        sim_label, len(top_matches),
    )

    # ── Google Fact-Check evidence ────────────────────────────────────────────
    google_label = api_results.get("google", "Unknown")
    evidence["factcheck_evidence"] = {
        "label":       google_label,
        "source":      "Google Fact Check Tools API",
        "description": (
            "Cross-references the claim against a global database of verified "
            "fact-check articles published by registered fact-checkers worldwide."
        ),
    }

    # ── NewsAPI evidence ──────────────────────────────────────────────────────
    news_label   = api_results.get("news", "Unknown")
    article_list = _summarise_articles(news_articles or [])
    evidence["newsapi_evidence"] = {
        "label":         news_label,
        "article_count": len(news_articles) if news_articles else 0,
        "articles":      article_list,          # top-5 article summaries
        "description":   (
            "Searches NewsAPI.org for recent news articles related to the claim "
            "and checks coverage by trusted international news outlets."
        ),
    }

    # ── Groq LLM evidence ──────────────────────────────────────────────────────
    groq_label = api_results.get("groq", "Unknown")
    evidence["groq_evidence"] = {
        "label":       groq_label,
        "source":      "Groq LLM (Llama 3.3 70B)",
        "description": (
            "Evaluates claim authenticity using Groq's high-speed Llama 3.3 70B reasoning model."
        ),
    }

    # ── Source credibility evidence ───────────────────────────────────────────
    if source_credibility:
        evidence["source_evidence"] = {
            "source_score":  source_credibility.get("source_score",  0.0),
            "sources_found": source_credibility.get("sources_found", []),
            "description":   (
                "Average credibility of the news domains cited by API results, "
                "scored from 0.0 (low credibility) to 1.0 (highest credibility)."
            ),
        }
    else:
        evidence["source_evidence"] = {
            "source_score":  None,
            "sources_found": [],
            "description":   "Source credibility data not available.",
        }

    logger.info(
        "evidence_builder: built evidence for claim='%.60s...' with %d sections.",
        claim, len(evidence),
    )
    return evidence


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────

def _summarise_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Extract a compact summary (title, source, url) from raw NewsAPI articles.

    Returns up to 5 articles with only display-friendly fields.
    """
    summaries: List[Dict[str, str]] = []
    for article in articles[:5]:
        source_name = (
            article.get("source", {}).get("name", "")
            or article.get("source", {}).get("id", "")
            or "Unknown"
        )
        url = article.get("url", "")
        domain = _extract_domain(url)
        summaries.append({
            "title":  (article.get("title") or "")[:120],
            "source": source_name,
            "domain": domain,
            "url":    url,
        })
    return summaries


def _extract_domain(url: str) -> str:
    """Extract the domain (e.g. 'reuters.com') from a URL string."""
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        # Strip leading www.
        return hostname.replace("www.", "").lower()
    except Exception:
        return ""
