"""
orchestration/analysis_orchestrator.py
----------------------------------------
Main entry point for the forensic analysis integration layer.

Public API
----------
    analyze_evidence(file_path, submitted_by="system") → ForensicReport

This function:
1. Creates an EvidenceObject (generates evidence_id, computes SHA-256).
2. Routes the file through evidence_router.py to determine applicable modules.
3. Calls each applicable adapter in sequence.
4. ONE module failure NEVER crashes the rest — each is wrapped in try/except.
5. Saves the final JSON report to reports/<evidence_id>.json.
6. Returns a ForensicReport.

Design principles enforced here
--------------------------------
- No hardcoded paths.
- No audio support.
- No Supabase (local storage only).
- No model retraining.
- Evidence ID is generated once and passed to all modules.
"""

import json
import logging
import time
from pathlib import Path

from config.settings import REPORTS_DIR, configure_logging
from schemas.evidence import EvidenceObject
from schemas.results import ModuleResult, ForensicReport
from orchestration.evidence_router import route_evidence, describe_routing, RoutingDecision
from orchestration.result_aggregator import aggregate_results

# ── Module adapters (imported lazily per routing decision) ─────────────────────
# We import at function call time to avoid loading 746 MB of model weights
# on server startup when only a subset of modules will be used.

logger = logging.getLogger(__name__)


def analyze_evidence(
    file_path: str,
    submitted_by: str = "system",
    original_filename: str | None = None,
) -> ForensicReport:
    """
    Full forensic analysis pipeline for a single evidence file.

    Parameters
    ----------
    file_path : str
        Absolute (or resolvable) path to the evidence file.
    submitted_by : str
        Name or ID of the submitting user for custody logging.
    original_filename : str | None
        Preserves original client filename if uploading via temporary API path.

    Returns
    -------
    ForensicReport
        Complete report including all module results and overall_status.
    """
    wall_t0 = time.perf_counter()
    path = Path(file_path).resolve()

    # ── 1. Pre-flight checks ───────────────────────────────────────────────────
    if not path.exists():
        raise FileNotFoundError(f"Evidence file not found: {path}")

    decision: RoutingDecision = route_evidence(str(path))
    logger.info("[%s] Routing: %s", path.name, describe_routing(decision))

    if not decision.is_supported:
        raise ValueError(decision.reject_reason)

    # ── 2. Create evidence object (generates evidence_id + SHA-256) ────────────
    evidence = EvidenceObject(
        file_path=path,
        file_type=decision.file_type,
        submitted_by=submitted_by,
    )
    if original_filename:
        evidence.file_name = original_filename
    eid = evidence.evidence_id
    logger.info(
        "[%s] evidence_id=%s  sha256=%s…",
        path.name, eid, evidence.sha256[:16],
    )

    # ── 3. Run blockchain (always first — registers evidence + hash) ───────────
    blockchain_result: ModuleResult | None = None
    if decision.run_blockchain:
        try:
            from integrations.blockchain.adapter import register_evidence
            blockchain_result = register_evidence(
                evidence_id=eid,
                file_path=str(path),
                sha256=evidence.sha256,
                file_type=decision.file_type,
                file_name=evidence.file_name,
                file_size=evidence.size_bytes,
                user=submitted_by,
            )
            logger.info(
                "[%s][blockchain] status=%s",
                eid, blockchain_result.status
            )
        except Exception as exc:
            logger.error("[%s][blockchain] UNHANDLED: %s", eid, exc, exc_info=True)
            blockchain_result = ModuleResult.failed(
                evidence_id=eid, analysis_type="blockchain", error=str(exc)
            )

    # ── 4. Run metadata forensics ──────────────────────────────────────────────
    metadata_result: ModuleResult | None = None
    if decision.run_metadata:
        try:
            from integrations.metadata.adapter import analyze_metadata
            metadata_result = analyze_metadata(
                evidence_id=eid,
                file_path=str(path),
                file_type=decision.file_type,
            )
            logger.info(
                "[%s][metadata] status=%s", eid, metadata_result.status
            )
        except Exception as exc:
            logger.error("[%s][metadata] UNHANDLED: %s", eid, exc, exc_info=True)
            metadata_result = ModuleResult.failed(
                evidence_id=eid, analysis_type="metadata_forensics", error=str(exc)
            )

    # ── 5. Run image forgery detection ────────────────────────────────────────
    image_forgery_result: ModuleResult | None = None
    if decision.run_image_forgery:
        try:
            from integrations.image_forgery.adapter import analyze_image_forgery
            image_forgery_result = analyze_image_forgery(
                evidence_id=eid,
                file_path=str(path),
                file_type=decision.file_type,
            )
            logger.info(
                "[%s][image_forgery] status=%s", eid, image_forgery_result.status
            )
        except Exception as exc:
            logger.error("[%s][image_forgery] UNHANDLED: %s", eid, exc, exc_info=True)
            image_forgery_result = ModuleResult.failed(
                evidence_id=eid, analysis_type="image_forgery", error=str(exc)
            )

    # ── 6. Run deepfake detection ──────────────────────────────────────────────
    deepfake_result: ModuleResult | None = None
    if decision.run_deepfake:
        try:
            from integrations.deepfake.adapter import analyze_deepfake
            deepfake_result = analyze_deepfake(
                evidence_id=eid,
                file_path=str(path),
                file_type=decision.file_type,
            )
            logger.info(
                "[%s][deepfake] status=%s", eid, deepfake_result.status
            )
        except Exception as exc:
            logger.error("[%s][deepfake] UNHANDLED: %s", eid, exc, exc_info=True)
            deepfake_result = ModuleResult.failed(
                evidence_id=eid, analysis_type="deepfake_detection", error=str(exc)
            )

    # ── 7. Run fake news detection (conditional) ───────────────────────────────
    fake_news_result: ModuleResult | None = None
    if decision.run_fake_news:
        try:
            from integrations.fake_news.adapter import (
                is_news_content_applicable,
                analyze_fake_news,
            )
            if is_news_content_applicable(str(path), decision.file_type):
                fake_news_result = analyze_fake_news(
                    evidence_id=eid,
                    file_path=str(path),
                    file_type=decision.file_type,
                )
                logger.info(
                    "[%s][fake_news] status=%s", eid, fake_news_result.status
                )
            else:
                fake_news_result = ModuleResult.skipped(
                    evidence_id=eid,
                    analysis_type="fake_news_detection",
                    reason="no_news_content_applicable",
                )
        except Exception as exc:
            logger.error("[%s][fake_news] UNHANDLED: %s", eid, exc, exc_info=True)
            fake_news_result = ModuleResult.failed(
                evidence_id=eid, analysis_type="fake_news_detection", error=str(exc)
            )

    # ── 8. Assemble final report ───────────────────────────────────────────────
    report = aggregate_results(
        evidence_dict=evidence.to_dict(),
        blockchain_result=blockchain_result,
        metadata_result=metadata_result,
        image_forgery_result=image_forgery_result,
        deepfake_result=deepfake_result,
        fake_news_result=fake_news_result,
    )

    elapsed = time.perf_counter() - wall_t0
    logger.info(
        "[%s] ANALYSIS COMPLETE in %.2fs — overall_status=%s",
        eid, elapsed, report.overall_status,
    )

    # ── 9. Save JSON report ────────────────────────────────────────────────────
    _save_report(report, eid)

    return report


def _save_report(report: ForensicReport, evidence_id: str) -> None:
    """Persist the ForensicReport to Supabase (if active) and as a local JSON file."""
    report_dict = report.to_dict()
    try:
        from database.db_service import save_report_record
        save_report_record(report_dict)
    except Exception as exc:
        logger.warning("Supabase save_report warning for %s: %s", evidence_id, exc)

    try:
        out_path = REPORTS_DIR / f"{evidence_id}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, default=str)
        logger.info("Report saved: %s", out_path)
    except Exception as exc:
        logger.error("Failed to save local report for %s: %s", evidence_id, exc)


def load_report(evidence_id: str) -> dict | None:
    """
    Load a previously saved ForensicReport from Supabase or local disk.

    Returns
    -------
    dict or None
        Parsed JSON dict, or None if the report file does not exist.
    """
    try:
        from database.db_service import get_report_record
        supa_report = get_report_record(evidence_id)
        if supa_report:
            return supa_report
    except Exception:
        pass

    report_path = REPORTS_DIR / f"{evidence_id}.json"
    if not report_path.exists():
        return None
    try:
        with report_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.error("Failed to load report %s: %s", evidence_id, exc)
        return None


def list_reports() -> list[str]:
    """Return a list of all evidence IDs that have saved reports."""
    try:
        return [p.stem for p in sorted(REPORTS_DIR.glob("EV-*.json"))]
    except Exception:
        return []
