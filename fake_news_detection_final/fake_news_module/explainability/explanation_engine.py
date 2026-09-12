"""
fake_news_module/explainability/explanation_engine.py
=======================================================
Phase 3 — Explainability Engine (orchestrator).

Combines evidence from every pipeline signal and generates:
  1. A list of human-readable reason strings explaining the verdict.
  2. A structured confidence breakdown (signal weights as percentages).
  3. A rich evidence dict for UI rendering.

This module is the ONLY import the pipeline needs from the explainability
sub-package — all complexity lives in evidence_builder.py and
confidence_breakdown.py.

Usage (internal)
----------------
    from fake_news_module.explainability.explanation_engine import generate_explanation

    explanation = generate_explanation(
        claim          = "5G towers cause COVID-19",
        verdict        = "Fake",
        confidence     = "High",
        score          = -0.82,
        api_results    = {"roberta": "Fake", "similarity": "Fake", ...},
        similarity_top_matches = [...],
        news_articles  = [...],
        source_credibility = {...},
    )
    # → {
    #     "verdict":              "Fake",
    #     "confidence":           "High",
    #     "confidence_pct":       82,
    #     "reasons":              ["RoBERTa classified...", ...],
    #     "evidence":             {...},
    #     "confidence_breakdown": {...},
    # }
"""

import logging
from typing import Any, Dict, List, Optional

from fake_news_module.explainability.evidence_builder import build_evidence
from fake_news_module.explainability.confidence_breakdown import build_confidence_breakdown

logger = logging.getLogger(__name__)


def generate_explanation(
    claim: str,
    verdict: str,
    confidence: str,
    score: float,
    api_results: Dict[str, str],
    similarity_top_matches: Optional[List[Dict[str, Any]]] = None,
    news_articles: Optional[List[Dict[str, Any]]] = None,
    source_credibility: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a full explainability report for a pipeline verdict.

    Args:
        claim:                 The extracted news claim (plain text).
        verdict:               Final decision string: "Real", "Fake", or "Uncertain".
        confidence:            Confidence tier: "High", "Medium", or "Low".
        score:                 Weighted aggregate score in [-1.0, +1.0].
        api_results:           Dict of {signal_name: label_string} from pipeline.
        similarity_top_matches: List of FAISS top-K match dicts (optional).
        news_articles:         Raw NewsAPI article list (optional).
        source_credibility:    Output from source_scorer module (optional).

    Returns:
        Dict with keys:
            verdict              : str
            confidence           : str
            confidence_pct       : int   — abs(score) mapped to 0–100%
            reasons              : List[str] — human-readable explanation sentences
            evidence             : Dict — structured signal evidence
            confidence_breakdown : Dict — per-signal weight percentages
    """
    logger.info(
        "explanation_engine: generating explanation for verdict='%s' score=%.4f",
        verdict, score,
    )

    # ── Build structured evidence ─────────────────────────────────────────────
    evidence = build_evidence(
        claim=claim,
        api_results=api_results,
        similarity_top_matches=similarity_top_matches,
        news_articles=news_articles,
        source_credibility=source_credibility,
    )

    # ── Build confidence breakdown ────────────────────────────────────────────
    source_score_val: Optional[float] = None
    if source_credibility and source_credibility.get("sources_found"):
        source_score_val = source_credibility.get("source_score")

    confidence_breakdown = build_confidence_breakdown(
        api_results=api_results,
        source_score=source_score_val,
    )

    # ── Build human-readable reasons ──────────────────────────────────────────
    reasons = _build_reasons(
        verdict=verdict,
        score=score,
        api_results=api_results,
        similarity_top_matches=similarity_top_matches or [],
        news_articles=news_articles or [],
        source_credibility=source_credibility,
        confidence_breakdown=confidence_breakdown,
    )

    # ── Map score to confidence percentage ───────────────────────────────────
    # abs(score) in [0,1] → percentage; clamp to [1, 100]
    confidence_pct = max(1, min(100, round(abs(score) * 100)))

    explanation: Dict[str, Any] = {
        "verdict":              verdict,
        "confidence":           confidence,
        "confidence_pct":       confidence_pct,
        "reasons":              reasons,
        "evidence":             evidence,
        "confidence_breakdown": confidence_breakdown,
    }

    logger.info(
        "explanation_engine: generated %d reason(s) for verdict='%s' "
        "confidence_pct=%d.",
        len(reasons), verdict, confidence_pct,
    )
    return explanation


# ──────────────────────────────────────────────────────────────────────────────
# Reason-sentence builders
# ──────────────────────────────────────────────────────────────────────────────

def _build_reasons(
    verdict: str,
    score: float,
    api_results: Dict[str, str],
    similarity_top_matches: List[Dict[str, Any]],
    news_articles: List[Dict[str, Any]],
    source_credibility: Optional[Dict[str, Any]],
    confidence_breakdown: Dict[str, Any],
) -> List[str]:
    """
    Compose a list of plain-English reason sentences, one per signal.

    Reasons are only added when a signal returned a *meaningful* (non-Unknown /
    non-Uncertain) result, so the list stays focused and avoids noise.
    """
    reasons: List[str] = []

    # 1. RoBERTa classifier ───────────────────────────────────────────────────
    roberta_label = api_results.get("roberta", "Unknown")
    roberta_info  = confidence_breakdown.get("roberta", {})
    if roberta_info.get("contributed", False):
        pct = roberta_info.get("weight_pct", 35)
        reasons.append(
            f"The local RoBERTa transformer model (carrying {pct}% weight) "
            f"classified this article as '{roberta_label}' based on "
            f"linguistic and semantic patterns learned from 44,898 labelled articles."
        )

    # 2. FAISS similarity search ──────────────────────────────────────────────
    sim_label = api_results.get("similarity", "Uncertain")
    sim_info  = confidence_breakdown.get("similarity", {})
    if sim_info.get("contributed", False) and similarity_top_matches:
        pct        = sim_info.get("weight_pct", 22)
        fake_count = sum(1 for m in similarity_top_matches if m.get("label") == "Fake")
        real_count = sum(1 for m in similarity_top_matches if m.get("label") == "Real")
        top_score  = similarity_top_matches[0].get("score", 0.0) if similarity_top_matches else 0.0
        reasons.append(
            f"Semantic similarity search (carrying {pct}% weight) found "
            f"{len(similarity_top_matches)} highly similar claims in the training corpus "
            f"(top similarity: {top_score:.2f}): {real_count} labelled Real and "
            f"{fake_count} labelled Fake — majority vote returned '{sim_label}'."
        )
    elif sim_info.get("contributed", False):
        reasons.append(
            f"Semantic similarity search returned '{sim_label}' based on "
            "nearest-neighbour retrieval from the 44,898-claim FAISS index."
        )

    # 3. Google Fact-Check API ────────────────────────────────────────────────
    google_label = api_results.get("google", "Unknown")
    google_info  = confidence_breakdown.get("google", {})
    if google_info.get("contributed", False):
        pct = google_info.get("weight_pct", 18)
        if google_label.lower() in {"fake", "false", "misleading"}:
            reasons.append(
                f"Google Fact Check Tools API (carrying {pct}% weight) found "
                f"matching claims that have been rated as '{google_label}' by "
                "verified international fact-checkers."
            )
        elif google_label.lower() in {"real", "true"}:
            reasons.append(
                f"Google Fact Check Tools API (carrying {pct}% weight) found "
                f"the claim was rated as '{google_label}' by registered fact-checkers."
            )

    # 4. NewsAPI ──────────────────────────────────────────────────────────────
    news_label = api_results.get("news", "Unknown")
    news_info  = confidence_breakdown.get("news", {})
    if news_info.get("contributed", False):
        pct           = news_info.get("weight_pct", 10)
        article_count = len(news_articles)
        if news_label == "Real" and article_count > 0:
            reasons.append(
                f"NewsAPI search (carrying {pct}% weight) retrieved {article_count} "
                "related articles, including coverage from multiple trusted "
                "international news outlets — supporting the claim's credibility."
            )
        elif news_label == "Uncertain":
            reasons.append(
                f"NewsAPI search (carrying {pct}% weight) retrieved {article_count} "
                "articles but found insufficient coverage from trusted sources "
                "to confirm or deny the claim."
            )

    # 5. Groq LLM API ─────────────────────────────────────────────────────────
    groq_label = api_results.get("groq", "Unknown")
    groq_info  = confidence_breakdown.get("groq", {})
    if groq_info.get("contributed", False):
        pct = groq_info.get("weight_pct", 25)
        if groq_label.lower() in {"true", "real"}:
            reasons.append(
                f"Groq LLM (carrying {pct}% weight) verified this claim as authentic based on its knowledge base."
            )
        elif groq_label.lower() in {"false", "fake"}:
            reasons.append(
                f"Groq LLM (carrying {pct}% weight) identified this claim as false or misleading based on factual analysis."
            )

    # 6. Source credibility ───────────────────────────────────────────────────
    source_info = confidence_breakdown.get("source", {})
    if source_credibility and source_info.get("contributed", False):
        src_score    = source_credibility.get("source_score", 0.0)
        src_found    = source_credibility.get("sources_found", [])
        src_label    = source_info.get("label", "Unknown")
        pct          = source_info.get("weight_pct", 10)
        domain_names = ", ".join(s.get("domain", "") for s in src_found[:3] if s.get("domain"))
        reasons.append(
            f"Source credibility analysis (carrying {pct}% weight) evaluated the "
            f"domains associated with API results and assigned an average credibility "
            f"score of {src_score:.2f}/1.00 ({src_label})"
            + (f" — sources found: {domain_names}." if domain_names else ".")
        )

    # 7. Fallback when no signal contributed ─────────────────────────────────
    if not reasons:
        reasons.append(
            "All verification signals returned Unknown or Uncertain responses. "
            "This may indicate missing API keys, a topic outside known databases, "
            "or very low-quality OCR text extraction. The verdict is based on a "
            f"weighted score of {score:+.4f}."
        )

    # 8. Overall summary sentence ─────────────────────────────────────────────
    summary = _build_summary(verdict, score, len(reasons))
    reasons.insert(0, summary)   # Put summary first

    return reasons


def _build_summary(verdict: str, score: float, signal_count: int) -> str:
    """Generate a one-sentence summary reason that leads the reasons list."""
    direction = "leaning towards Real" if score > 0 else "leaning towards Fake"
    if abs(score) > 0.5:
        strength = "strong"
    elif abs(score) > 0.2:
        strength = "moderate"
    else:
        strength = "weak"

    if verdict == "Real":
        return (
            f"The weighted analysis of {signal_count} verification signals produced "
            f"a {strength} consensus verdict of REAL (score: {score:+.4f}), "
            "indicating this content is likely authentic."
        )
    elif verdict == "Fake":
        return (
            f"The weighted analysis of {signal_count} verification signals produced "
            f"a {strength} consensus verdict of FAKE (score: {score:+.4f}), "
            "indicating this content exhibits characteristics of misinformation."
        )
    else:
        return (
            f"The weighted analysis of {signal_count} verification signals produced "
            f"an UNCERTAIN verdict (score: {score:+.4f}, {direction}). "
            "Insufficient evidence was available to make a definitive determination."
        )
