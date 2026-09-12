"""
custody.py
----------
Chain-of-custody logging and tamper alert management.

Every action on evidence is recorded:
  - upload   : File first registered in the system
  - verify   : Integrity check performed
  - approve  : Officer approved file for blockchain
  - access   : File record was accessed
  - alert    : Tampering was detected (saved to alerts table)
"""

from datetime import datetime
from database.db import get_connection

# Action type constants
ACTION_UPLOAD  = "upload"
ACTION_VERIFY  = "verify"
ACTION_APPROVE = "approve"
ACTION_ACCESS  = "access"
ACTION_ALERT   = "alert"


def log_custody_event(
    evidence_id: str,
    action: str,
    user: str = "system",
    notes: str = "",
) -> dict:
    """
    Record a custody event in the audit log.

    Args:
        evidence_id (str): Unique evidence identifier.
        action      (str): Action type — use ACTION_* constants.
        user        (str): Who performed the action.
        notes       (str): Optional detail note.

    Returns:
        dict: The recorded event.
    """
    if not evidence_id or not action:
        raise ValueError("evidence_id and action are required.")

    timestamp = datetime.utcnow().isoformat()
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO custody (evidence_id, action, user, timestamp, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (evidence_id, action, user, timestamp, notes),
        )
        conn.commit()
        row_id = cursor.lastrowid
    finally:
        conn.close()

    return {
        "id":          row_id,
        "evidence_id": evidence_id,
        "action":      action,
        "user":        user,
        "timestamp":   timestamp,
        "notes":       notes,
    }


def save_tamper_alert(
    evidence_id: str,
    file_name: str,
    file_type: str,
    stored_hash: str,
    current_hash: str,
    detected_by: str = "system",
) -> dict:
    """
    Save a tampering alert to the alerts table.

    Called automatically when verify_integrity() returns TAMPERED.
    This creates a permanent record of the tampering event.

    Args:
        evidence_id  (str): Evidence ID of the tampered file.
        file_name    (str): Name of the tampered file.
        file_type    (str): Type — image/video/audio/document.
        stored_hash  (str): Original hash stored at upload.
        current_hash (str): Hash computed during verification.
        detected_by  (str): Who/what triggered the verification.

    Returns:
        dict: The saved alert record.
    """
    timestamp = datetime.utcnow().isoformat()
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            INSERT INTO tamper_alerts
                (evidence_id, file_name, file_type, stored_hash,
                 current_hash, detected_by, timestamp, resolved)
            VALUES (?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                evidence_id, file_name, file_type,
                stored_hash, current_hash, detected_by, timestamp,
            ),
        )
        conn.commit()
        alert_id = cursor.lastrowid
    finally:
        conn.close()

    # Also log to custody trail
    log_custody_event(
        evidence_id = evidence_id,
        action      = ACTION_ALERT,
        user        = detected_by,
        notes       = f"TAMPER ALERT #{alert_id}: Hash mismatch detected.",
    )

    return {
        "alert_id":    alert_id,
        "evidence_id": evidence_id,
        "file_name":   file_name,
        "file_type":   file_type,
        "stored_hash": stored_hash,
        "current_hash":current_hash,
        "detected_by": detected_by,
        "timestamp":   timestamp,
        "resolved":    False,
    }


def get_all_alerts(resolved: bool = False) -> list[dict]:
    """
    Retrieve tamper alerts from the database.

    Args:
        resolved (bool): If True, return resolved alerts; else unresolved.

    Returns:
        list[dict]: All matching tamper alerts.
    """
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT id, evidence_id, file_name, file_type, stored_hash,
                   current_hash, detected_by, timestamp, resolved
            FROM tamper_alerts
            WHERE resolved = ?
            ORDER BY timestamp DESC
            """,
            (1 if resolved else 0,),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    return [
        {
            "alert_id":     row[0],
            "evidence_id":  row[1],
            "file_name":    row[2],
            "file_type":    row[3],
            "stored_hash":  row[4],
            "current_hash": row[5],
            "detected_by":  row[6],
            "timestamp":    row[7],
            "resolved":     bool(row[8]),
        }
        for row in rows
    ]


def get_custody_log(evidence_id: str) -> list[dict]:
    """
    Get the full custody history for one evidence item.

    Args:
        evidence_id (str): Evidence ID to query.

    Returns:
        list[dict]: All custody events, oldest first.
    """
    conn = get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT id, evidence_id, action, user, timestamp, notes
            FROM custody
            WHERE evidence_id = ?
            ORDER BY timestamp ASC
            """,
            (evidence_id,),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    return [
        {
            "id":          row[0],
            "evidence_id": row[1],
            "action":      row[2],
            "user":        row[3],
            "timestamp":   row[4],
            "notes":       row[5],
        }
        for row in rows
    ]
