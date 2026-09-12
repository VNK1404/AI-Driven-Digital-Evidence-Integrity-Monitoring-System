"""
database/db_service.py
-----------------------
Unified Database Service — AI-Driven Digital Evidence Integrity Monitoring System

Coordinates data persistence across Supabase (primary cloud DB when credentials set)
and SQLite (`blockchain_chain_of_custudy/database/evidence.db` local fallback).

Ensures complete system reliability: if Supabase is offline or unconfigured,
all operations complete seamlessly via local SQLite and local JSON files.
"""

import logging
from typing import Any
from config.settings import is_supabase_enabled
from database.supabase_client import (
    save_evidence_supabase,
    get_evidence_supabase,
    update_evidence_status_supabase,
    log_custody_event_supabase,
    get_custody_log_supabase,
    save_tamper_alert_supabase,
    save_approval_supabase,
    save_report_supabase,
    get_report_supabase,
)

logger = logging.getLogger(__name__)

# ── SQLite Fallback Imports ───────────────────────────────────────────────────

def _get_sqlite_db():
    from blockchain_chain_of_custudy.database.db import (  # type: ignore[import]
        save_evidence as sqlite_save_evidence,
        get_evidence as sqlite_get_evidence,
        update_evidence_status as sqlite_update_status,
        save_approval as sqlite_save_approval,
    )
    return sqlite_save_evidence, sqlite_get_evidence, sqlite_update_status, sqlite_save_approval


def _get_sqlite_custody():
    from blockchain_chain_of_custudy.integrity.custody import (  # type: ignore[import]
        log_custody_event as sqlite_log_event,
        get_custody_log as sqlite_get_log,
        save_tamper_alert as sqlite_save_alert,
    )
    return sqlite_log_event, sqlite_get_log, sqlite_save_alert


# ── Unified Public Interface ───────────────────────────────────────────────────

def save_evidence_record(
    evidence_id: str,
    file_name: str,
    file_path: str,
    file_type: str,
    file_size: int,
    hash_value: str,
    uploaded_by: str,
    upload_time: str,
) -> None:
    """Save evidence record to both Supabase (if active) and local SQLite."""
    record = {
        "evidence_id": evidence_id,
        "file_name":   file_name,
        "file_path":   file_path,
        "file_type":   file_type,
        "size_bytes":  file_size,
        "sha256":      hash_value,
        "submitted_by": uploaded_by,
        "submitted_at": upload_time,
        "status":      "pending",
    }

    # 1. Supabase (Primary Cloud Storage & DB)
    if is_supabase_enabled():
        from database.supabase_client import upload_file_to_supabase_storage
        storage_res = upload_file_to_supabase_storage(file_path, evidence_id, file_name)
        record["storage_path"] = storage_res.get("storage_path", "")
        record["storage_url"]  = storage_res.get("storage_url", "")

        ok = save_evidence_supabase(record)
        if ok:
            logger.info("[%s] Evidence record saved to Supabase.", evidence_id)

    # 2. Local SQLite (Always saved for local backup)
    try:
        sqlite_save_evidence, *_ = _get_sqlite_db()
        sqlite_save_evidence(
            evidence_id=evidence_id,
            file_name=file_name,
            file_path=file_path,
            file_type=file_type,
            file_size=file_size,
            hash_value=hash_value,
            uploaded_by=uploaded_by,
            upload_time=upload_time,
        )
    except Exception as exc:
        logger.warning("[%s] SQLite save_evidence warning: %s", evidence_id, exc)


def get_evidence_record(evidence_id: str) -> dict | None:
    """Fetch evidence record from Supabase first, falling back to local SQLite."""
    if is_supabase_enabled():
        supa_res = get_evidence_supabase(evidence_id)
        if supa_res:
            return supa_res

    try:
        _, sqlite_get_evidence, *_ = _get_sqlite_db()
        row = sqlite_get_evidence(evidence_id)
        return dict(row) if row else None
    except Exception as exc:
        logger.error("[%s] SQLite get_evidence failed: %s", evidence_id, exc)
        return None


def update_evidence_status_record(evidence_id: str, status: str) -> None:
    """Update evidence status in Supabase and local SQLite."""
    if is_supabase_enabled():
        update_evidence_status_supabase(evidence_id, status)

    try:
        _, _, sqlite_update_status, _ = _get_sqlite_db()
        sqlite_update_status(evidence_id, status)
    except Exception as exc:
        logger.warning("[%s] SQLite update_status warning: %s", evidence_id, exc)


def log_custody_event_record(
    evidence_id: str,
    action: str,
    user: str = "system",
    notes: str = "",
    timestamp: str | None = None,
) -> None:
    """Record chain-of-custody event in Supabase and local SQLite."""
    from datetime import datetime, timezone
    ts = timestamp or datetime.now(timezone.utc).isoformat()

    if is_supabase_enabled():
        log_custody_event_supabase(evidence_id, action, user, notes, ts)

    try:
        sqlite_log_event, *_ = _get_sqlite_custody()
        sqlite_log_event(evidence_id=evidence_id, action=action, user=user, notes=notes)
    except Exception as exc:
        logger.warning("[%s] SQLite custody log warning: %s", evidence_id, exc)


def get_custody_log_records(evidence_id: str) -> list[dict]:
    """Retrieve custody history from Supabase first, falling back to SQLite."""
    if is_supabase_enabled():
        supa_log = get_custody_log_supabase(evidence_id)
        if supa_log:
            return supa_log

    try:
        _, sqlite_get_log, _ = _get_sqlite_custody()
        return sqlite_get_log(evidence_id)
    except Exception as exc:
        logger.error("[%s] SQLite get_custody_log failed: %s", evidence_id, exc)
        return []


def save_tamper_alert_record(alert_dict: dict) -> None:
    """Save tamper alert to Supabase and local SQLite."""
    if is_supabase_enabled():
        save_tamper_alert_supabase(alert_dict)

    try:
        _, _, sqlite_save_alert = _get_sqlite_custody()
        sqlite_save_alert(
            evidence_id=alert_dict["evidence_id"],
            file_name=alert_dict["file_name"],
            file_type=alert_dict["file_type"],
            stored_hash=alert_dict["stored_hash"],
            current_hash=alert_dict["current_hash"],
            detected_by=alert_dict.get("detected_by", "system"),
        )
    except Exception as exc:
        logger.warning("[%s] SQLite save_tamper_alert warning: %s", alert_dict.get("evidence_id"), exc)


def save_approval_record(
    evidence_id: str,
    approved_by: str,
    block_index: int,
    timestamp: str,
    notes: str = "",
) -> None:
    """Save officer blockchain approval to Supabase and local SQLite."""
    if is_supabase_enabled():
        save_approval_supabase(evidence_id, approved_by, block_index, timestamp, notes)

    try:
        *_, sqlite_save_approval = _get_sqlite_db()
        sqlite_save_approval(
            evidence_id=evidence_id,
            approved_by=approved_by,
            block_index=block_index,
            timestamp=timestamp,
            notes=notes,
        )
    except Exception as exc:
        logger.warning("[%s] SQLite save_approval warning: %s", evidence_id, exc)


def save_report_record(report_dict: dict) -> None:
    """Persist full ForensicReport payload to Supabase."""
    eid = report_dict.get("evidence_id")
    if is_supabase_enabled() and eid:
        ok = save_report_supabase(report_dict)
        if ok:
            logger.info("[%s] Forensic report persisted to Supabase.", eid)


def get_report_record(evidence_id: str) -> dict | None:
    """Fetch ForensicReport payload from Supabase if available."""
    if is_supabase_enabled():
        return get_report_supabase(evidence_id)
    return None
