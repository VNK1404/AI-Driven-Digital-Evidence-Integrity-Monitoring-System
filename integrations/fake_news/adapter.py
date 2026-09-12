"""
integrations/fake_news/adapter.py
-----------------------------------
Adapter for the fake_news_detection_final module.

Key design decisions
---------------------
1. fake_news_pipeline() is imported from the existing module via sys.path
   injection into fake_news_detection_final/.
2. Fake news detection ONLY applies to images and PDFs — NOT to every document
   or video.  The adapter provides is_news_content_applicable() which the
   orchestrator calls to decide whether to run the pipeline.
3. The pipeline calls asyncio.run() internally for external APIs.  This is
   safe in a synchronous Flask thread.
4. The .env file in fake_news_detection_final/ is automatically loaded by
   config.settings (dotenv) — no duplicate loading needed.
5. If the pipeline raises ValueError (no text extracted, bad quality), the
   adapter returns status='skipped' instead of 'failed' to distinguish a
   "no news content" outcome from a genuine error.

Supported file types for fake_news: image, document (PDF)
Skipped file types: video (fake_news makes no sense for video)
"""

import sys
import time
import logging
from pathlib import Path

from config.settings import FAKE_NEWS_ROOT
from schemas.results import ModuleResult

logger = logging.getLogger(__name__)

# ── Supported extensions for fake_news analysis ────────────────────────────────
_APPLICABLE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",  # images
    ".pdf",                                                       # documents
}

# ── sys.path injection ─────────────────────────────────────────────────────────
_fake_news_added = False


def _ensure_path():
    global _fake_news_added
    if not _fake_news_added:
        root = str(FAKE_NEWS_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        _fake_news_added = True


def _get_pipeline():
    _ensure_path()
    from fake_news_module.core.pipeline import (  # type: ignore[import]
        fake_news_pipeline,
        analyze_text_pipeline,
    )
    return fake_news_pipeline, analyze_text_pipeline


# ── Public API ─────────────────────────────────────────────────────────────────

def is_news_content_applicable(file_path: str, file_type: str) -> bool:
    """
    Determine whether fake_news analysis should be attempted on this file.

    Rules
    -----
    - Video files: NEVER — video has no text content for news analysis.
    - Image files: YES  — OCR may extract news text.
    - PDF files:   YES  — pdfplumber extracts text directly.
    - Other:       NO.
    """
    ext = Path(file_path).suffix.lower()
    return ext in _APPLICABLE_EXTENSIONS


def analyze_fake_news(
    evidence_id: str,
    file_path: str,
    file_type: str,
) -> ModuleResult:
    """
    Run the fake news detection pipeline on an image or PDF file.

    Parameters
    ----------
    evidence_id : str
        Shared evidence ID.
    file_path : str
        Absolute path to the image or PDF.
    file_type : str
        'image' or 'document'.  'video' returns status='skipped'.

    Returns
    -------
    ModuleResult
        status = 'success'  → fake_news analysis completed
        status = 'skipped'  → file type not applicable / no text extracted
        status = 'failed'   → unexpected pipeline error
    """
    # ── Guard: check applicability ─────────────────────────────────────────────
    if not is_news_content_applicable(str(file_path), file_type):
        return ModuleResult.skipped(
            evidence_id=evidence_id,
            analysis_type="fake_news_detection",
            reason=f"fake_news not applicable for file_type='{file_type}'",
        )

    t0 = time.perf_counter()
    try:
        pipeline_fn, _ = _get_pipeline()
        raw = pipeline_fn(str(file_path))

        result = {
            "final_decision": raw.get("final_decision", "Unknown"),
            "confidence":     raw.get("confidence", "Unknown"),
            "score":          round(float(raw.get("score", 0.0)), 4),
            "claim":          raw.get("claim", ""),
            "extracted_text_preview": (raw.get("extracted_text", "")[:500]
                                       if raw.get("extracted_text") else ""),
            "api_results":    raw.get("api_results", {}),
            "source_score":   raw.get("source_score", None),
            "explanation":    raw.get("explanation", ""),
        }

        return ModuleResult(
            evidence_id=evidence_id,
            analysis_type="fake_news_detection",
            status="success",
            duration_seconds=time.perf_counter() - t0,
            result=result,
        )

    except ValueError as exc:
        # Pipeline raises ValueError when OCR returns empty / text quality fails
        logger.info(
            "fake_news pipeline returned no usable text for %s: %s",
            evidence_id, exc,
        )
        return ModuleResult.skipped(
            evidence_id=evidence_id,
            analysis_type="fake_news_detection",
            reason=f"no_news_content_detected: {exc}",
        )
    except Exception as exc:
        logger.error(
            "fake_news.analyze_fake_news failed for %s: %s",
            evidence_id, exc, exc_info=True,
        )
        return ModuleResult.failed(
            evidence_id=evidence_id,
            analysis_type="fake_news_detection",
            error=str(exc),
            duration=time.perf_counter() - t0,
        )


def analyze_fake_news_text(
    evidence_id: str,
    text: str,
) -> ModuleResult:
    """
    Run the fake news pipeline on pre-extracted text (no OCR step).

    Useful when the caller has already extracted text from a document.
    """
    t0 = time.perf_counter()
    try:
        _, text_pipeline = _get_pipeline()
        raw = text_pipeline(text)

        result = {
            "final_decision": raw.get("final_decision", "Unknown"),
            "confidence":     raw.get("confidence", "Unknown"),
            "score":          round(float(raw.get("score", 0.0)), 4),
            "claim":          raw.get("claim", ""),
            "api_results":    raw.get("api_results", {}),
            "source_score":   raw.get("source_score", None),
            "explanation":    raw.get("explanation", ""),
        }

        return ModuleResult(
            evidence_id=evidence_id,
            analysis_type="fake_news_detection",
            status="success",
            duration_seconds=time.perf_counter() - t0,
            result=result,
        )

    except ValueError as exc:
        return ModuleResult.skipped(
            evidence_id=evidence_id,
            analysis_type="fake_news_detection",
            reason=f"text_analysis_skipped: {exc}",
        )
    except Exception as exc:
        logger.error(
            "fake_news.analyze_fake_news_text failed for %s: %s",
            evidence_id, exc, exc_info=True,
        )
        return ModuleResult.failed(
            evidence_id=evidence_id,
            analysis_type="fake_news_detection",
            error=str(exc),
            duration=time.perf_counter() - t0,
        )
