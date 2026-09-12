"""
integrations/blockchain/adapter.py
------------------------------------
Adapter for the blockchain_chain_of_custudy module.

Responsibilities
----------------
1. Inject blockchain_chain_of_custudy/ onto sys.path so its bare
   imports resolve correctly.
2. Expose a clean, evidence_id-aware interface:
       register_evidence(...)
       verify_integrity(...)
       log_custody_event(...)
       approve_evidence(...)
       get_chain_status()
       get_custody_log(evidence_id)
3. Persist data to Supabase (primary) and SQLite (backup) via database/db_service.py.
4. Return ModuleResult from every public method.
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

from config.settings import BLOCKCHAIN_ROOT
from schemas.results import ModuleResult
from database.db_service import (
    save_evidence_record,
    get_evidence_record,
    update_evidence_status_record,
    log_custody_event_record,
    get_custody_log_records,
    save_tamper_alert_record,
    save_approval_record,
)

logger = logging.getLogger(__name__)

# ── One-time sys.path injection ────────────────────────────────────────────────
_blockchain_added = False


def _ensure_path():
    global _blockchain_added
    if not _blockchain_added:
        root = str(BLOCKCHAIN_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        _blockchain_added = True


def _get_verify():
    _ensure_path()
    from integrity.verify import verify_integrity  # type: ignore[import]
    return verify_integrity


def _get_blockchain():
    _ensure_path()
    from integrity.blockchain import evidence_blockchain  # type: ignore[import]
    return evidence_blockchain


# ── Public API ─────────────────────────────────────────────────────────────────

def register_evidence(
    evidence_id: str,
    file_path: str,
    sha256: str,
    file_type: str,
    file_name: str,
    file_size: int,
    user: str = "system",
) -> ModuleResult:
    """
    Register a new evidence file in the custody system (Supabase + SQLite).
    """
    t0 = time.perf_counter()
    try:
        upload_time = datetime.now(timezone.utc).isoformat()

        save_evidence_record(
            evidence_id=evidence_id,
            file_name=file_name,
            file_path=str(file_path),
            file_type=file_type,
            file_size=file_size,
            hash_value=sha256,
            uploaded_by=user,
            upload_time=upload_time,
        )

        log_custody_event_record(
            evidence_id=evidence_id,
            action="upload",
            user=user,
            notes=f"Evidence registered: {file_name} ({file_type})",
            timestamp=upload_time,
        )

        chain = _get_blockchain()

        return ModuleResult(
            evidence_id=evidence_id,
            analysis_type="blockchain",
            status="success",
            duration_seconds=time.perf_counter() - t0,
            result={
                "sha256":           sha256,
                "registered_at":    upload_time,
                "integrity_status": "REGISTERED",
                "chain_length":     len(chain.chain),
                "chain_valid":      chain.is_chain_valid(),
            },
        )

    except Exception as exc:
        logger.error("blockchain.register_evidence failed: %s", exc, exc_info=True)
        return ModuleResult.failed(
            evidence_id=evidence_id,
            analysis_type="blockchain",
            error=str(exc),
            duration=time.perf_counter() - t0,
        )


def verify_integrity(
    evidence_id: str,
    file_path: str,
) -> ModuleResult:
    """
    Verify evidence file integrity by comparing its current SHA-256 against stored hash.
    """
    t0 = time.perf_counter()
    try:
        verify_fn = _get_verify()
        record = get_evidence_record(evidence_id)
        if record is None:
            raise ValueError(f"No evidence record found for ID: {evidence_id}")

        stored_hash = record.get("hash_value") or record.get("sha256") or ""
        file_name   = record.get("file_name", Path(file_path).name)
        file_type   = record.get("file_type", "image")

        verify_result = verify_fn(str(file_path), stored_hash)
        status = verify_result["status"]           # VERIFIED | TAMPERED | ERROR

        if status == "VERIFIED":
            update_evidence_status_record(evidence_id, "verified")
        elif status == "TAMPERED":
            update_evidence_status_record(evidence_id, "tampered")
            ts = datetime.now(timezone.utc).isoformat()
            save_tamper_alert_record({
                "evidence_id":  evidence_id,
                "file_name":    file_name,
                "file_type":    file_type,
                "stored_hash":  stored_hash,
                "current_hash": verify_result.get("current_hash", ""),
                "detected_by":  "forensic_system",
                "timestamp":    ts,
                "resolved":     False,
            })

        log_custody_event_record(
            evidence_id=evidence_id,
            action="verify",
            user="forensic_system",
            notes=f"Integrity check: {status}",
        )

        return ModuleResult(
            evidence_id=evidence_id,
            analysis_type="blockchain",
            status="success",
            duration_seconds=time.perf_counter() - t0,
            result={
                "sha256":           verify_result.get("current_hash", stored_hash),
                "stored_hash":      stored_hash,
                "integrity_status": status,
                "tampered":         verify_result.get("tampered", False),
                "message":          verify_result.get("message", ""),
            },
        )

    except Exception as exc:
        logger.error("blockchain.verify_integrity failed: %s", exc, exc_info=True)
        return ModuleResult.failed(
            evidence_id=evidence_id,
            analysis_type="blockchain",
            error=str(exc),
            duration=time.perf_counter() - t0,
        )


def log_custody_event(
    evidence_id: str,
    action: str,
    user: str = "system",
    notes: str = "",
) -> None:
    """Append a custody event to the audit log."""
    try:
        log_custody_event_record(evidence_id=evidence_id, action=action, user=user, notes=notes)
    except Exception as exc:
        logger.warning("custody log event failed (non-fatal): %s", exc)


def approve_evidence(
    evidence_id: str,
    approved_by: str = "officer",
    notes: str = "",
) -> ModuleResult:
    """
    Officer approval — adds evidence to blockchain ledger and records approval.
    """
    t0 = time.perf_counter()
    try:
        record = get_evidence_record(evidence_id)
        if record is None:
            raise ValueError(f"No evidence record found for ID: {evidence_id}")

        chain = _get_blockchain()
        block = chain.add_approved_block(
            evidence_data={
                "evidence_id": evidence_id,
                "file_name":   record.get("file_name", ""),
                "file_type":   record.get("file_type", ""),
                "hash_value":  record.get("hash_value") or record.get("sha256", ""),
                "file_size":   record.get("file_size") or record.get("size_bytes", 0),
            },
            approved_by=approved_by,
        )

        ts = datetime.now(timezone.utc).isoformat()
        save_approval_record(
            evidence_id=evidence_id,
            approved_by=approved_by,
            block_index=block.index,
            timestamp=ts,
            notes=notes,
        )

        update_evidence_status_record(evidence_id, "approved")

        log_custody_event_record(
            evidence_id=evidence_id,
            action="approve",
            user=approved_by,
            notes=f"Block #{block.index} — {notes}",
            timestamp=ts,
        )

        return ModuleResult(
            evidence_id=evidence_id,
            analysis_type="blockchain",
            status="success",
            duration_seconds=time.perf_counter() - t0,
            result={
                "block_index":  block.index,
                "block_hash":   block.hash,
                "approved_by":  approved_by,
                "approved_at":  ts,
                "chain_length": len(chain.chain),
                "chain_valid":  chain.is_chain_valid(),
            },
        )

    except Exception as exc:
        logger.error("blockchain.approve_evidence failed: %s", exc, exc_info=True)
        return ModuleResult.failed(
            evidence_id=evidence_id,
            analysis_type="blockchain",
            error=str(exc),
            duration=time.perf_counter() - t0,
        )


def get_chain_status() -> dict:
    """Return current state of the blockchain ledger."""
    try:
        chain = _get_blockchain()
        return {
            "chain_length": len(chain.chain),
            "chain_valid":  chain.is_chain_valid(),
            "chain":        chain.get_chain(),
        }
    except Exception as exc:
        logger.error("blockchain.get_chain_status failed: %s", exc)
        return {"chain_length": 0, "chain_valid": False, "error": str(exc)}


def get_custody_log(evidence_id: str) -> list:
    """Return full custody history for one evidence item."""
    try:
        return get_custody_log_records(evidence_id)
    except Exception as exc:
        logger.error("blockchain.get_custody_log failed: %s", exc)
        return []
