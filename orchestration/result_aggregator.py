"""
orchestration/result_aggregator.py
------------------------------------
Builds the final ForensicReport from individual ModuleResult objects.

Determines the overall_status:
  - 'completed'  → all scheduled modules succeeded or were intentionally skipped
  - 'partial'    → at least one scheduled module failed, but others succeeded
  - 'failed'     → all scheduled modules failed
"""

import logging
from schemas.results import ModuleResult, ForensicReport

logger = logging.getLogger(__name__)


def aggregate_results(
    evidence_dict: dict,
    blockchain_result: ModuleResult | None,
    metadata_result: ModuleResult | None,
    image_forgery_result: ModuleResult | None,
    deepfake_result: ModuleResult | None,
    fake_news_result: ModuleResult | None,
) -> ForensicReport:
    """
    Combine all module results into a single ForensicReport.

    Parameters
    ----------
    evidence_dict : dict
        EvidenceObject.to_dict() — the file metadata.
    blockchain_result, metadata_result, ... : ModuleResult | None
        Results from each adapter, or None if the module was not scheduled.

    Returns
    -------
    ForensicReport
        Complete report with overall_status set.
    """
    scheduled = [r for r in [
        blockchain_result,
        metadata_result,
        image_forgery_result,
        deepfake_result,
        fake_news_result,
    ] if r is not None]

    # Modules that were actually scheduled (not just None / unscheduled)
    ran = [r for r in scheduled if r.status != "skipped"]
    failed = [r for r in ran if r.status == "failed"]
    succeeded = [r for r in ran if r.status == "success"]

    if not ran:
        # All modules were skipped — unusual but possible
        overall = "completed"
    elif len(failed) == len(ran):
        overall = "failed"
    elif failed:
        overall = "partial"
    else:
        overall = "completed"

    if failed:
        logger.warning(
            "evidence_id=%s — %d module(s) failed: %s",
            evidence_dict.get("evidence_id"),
            len(failed),
            [r.analysis_type for r in failed],
        )

    return ForensicReport(
        evidence=evidence_dict,
        blockchain=blockchain_result,
        metadata=metadata_result,
        image_forgery=image_forgery_result,
        deepfake=deepfake_result,
        fake_news=fake_news_result,
        overall_status=overall,
    )
