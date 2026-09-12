"""
db.py
-----
SQLite database setup for the Evidence Integrity System.

Tables:
  evidence       — All registered files (any type)
  custody        — Immutable audit log of every action
  tamper_alerts  — Records of detected tampering events
  approvals      — Records of officer approvals for blockchain
"""

import sqlite3
import os

_DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_DB_DIR, "evidence.db")


def get_connection() -> sqlite3.Connection:
    """Open and return a SQLite connection with Row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Create all database tables if they don't exist.
    Safe to call multiple times — fully idempotent.

    Tables
    ------
    evidence:
        evidence_id   TEXT PK   – unique ID e.g. "EV001"
        file_name     TEXT      – original filename
        file_path     TEXT      – path at upload time
        file_type     TEXT      – image/video/audio/document
        file_size     INTEGER   – size in bytes
        hash_value    TEXT      – SHA-256 hex digest
        uploaded_by   TEXT      – who uploaded it
        upload_time   TEXT      – UTC ISO timestamp
        status        TEXT      – pending/verified/approved/tampered

    custody:
        id            INT PK    – auto increment
        evidence_id   TEXT      – ref to evidence
        action        TEXT      – upload/verify/approve/access/alert
        user          TEXT      – who did it
        timestamp     TEXT      – UTC ISO timestamp
        notes         TEXT      – optional detail

    tamper_alerts:
        id            INT PK    – auto increment
        evidence_id   TEXT      – which file was tampered
        file_name     TEXT      – filename for quick reference
        file_type     TEXT      – image/video/audio/document
        stored_hash   TEXT      – original hash
        current_hash  TEXT      – hash at time of detection
        detected_by   TEXT      – who ran the verify
        timestamp     TEXT      – when detected
        resolved      INTEGER   – 0=open, 1=resolved

    approvals:
        id            INT PK    – auto increment
        evidence_id   TEXT      – which file was approved
        approved_by   TEXT      – officer who approved
        block_index   INTEGER   – blockchain block position
        timestamp     TEXT      – when approved
        notes         TEXT      – optional reason/notes
    """
    conn = get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS evidence (
                evidence_id  TEXT    PRIMARY KEY,
                file_name    TEXT    NOT NULL,
                file_path    TEXT    NOT NULL,
                file_type    TEXT    NOT NULL,
                file_size    INTEGER NOT NULL,
                hash_value   TEXT    NOT NULL,
                uploaded_by  TEXT    NOT NULL DEFAULT 'system',
                upload_time  TEXT    NOT NULL,
                status       TEXT    NOT NULL DEFAULT 'pending'
            );

            CREATE TABLE IF NOT EXISTS custody (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                evidence_id  TEXT    NOT NULL,
                action       TEXT    NOT NULL,
                user         TEXT    NOT NULL,
                timestamp    TEXT    NOT NULL,
                notes        TEXT    DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS tamper_alerts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                evidence_id  TEXT    NOT NULL,
                file_name    TEXT    NOT NULL,
                file_type    TEXT    NOT NULL,
                stored_hash  TEXT    NOT NULL,
                current_hash TEXT    NOT NULL,
                detected_by  TEXT    NOT NULL,
                timestamp    TEXT    NOT NULL,
                resolved     INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS approvals (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                evidence_id  TEXT    NOT NULL,
                approved_by  TEXT    NOT NULL,
                block_index  INTEGER NOT NULL,
                timestamp    TEXT    NOT NULL,
                notes        TEXT    DEFAULT ''
            );
        """)
        conn.commit()
    finally:
        conn.close()


# ── Evidence CRUD ─────────────────────────────────────────────────────────────

def save_evidence(
    evidence_id: str,
    file_name: str,
    file_path: str,
    file_type: str,
    file_size: int,
    hash_value: str,
    uploaded_by: str,
    upload_time: str,
) -> None:
    """Insert a new evidence record. Status defaults to 'pending'."""
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO evidence
                (evidence_id, file_name, file_path, file_type,
                 file_size, hash_value, uploaded_by, upload_time, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')
            """,
            (evidence_id, file_name, file_path, file_type,
             file_size, hash_value, uploaded_by, upload_time),
        )
        conn.commit()
    finally:
        conn.close()


def update_evidence_status(evidence_id: str, status: str) -> None:
    """
    Update the status of an evidence record.

    Valid statuses: pending → verified → approved | tampered
    """
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE evidence SET status = ? WHERE evidence_id = ?",
            (status, evidence_id),
        )
        conn.commit()
    finally:
        conn.close()


def get_evidence(evidence_id: str) -> dict | None:
    """Fetch a single evidence record by ID."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM evidence WHERE evidence_id = ?",
            (evidence_id,),
        )
        row = cursor.fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def get_all_evidence(file_type: str = None, status: str = None) -> list[dict]:
    """
    Return all evidence records, optionally filtered.

    Args:
        file_type (str): Filter by 'image','video','audio','document'.
        status    (str): Filter by 'pending','verified','approved','tampered'.
    """
    query  = "SELECT * FROM evidence WHERE 1=1"
    params = []

    if file_type:
        query  += " AND file_type = ?"
        params.append(file_type)
    if status:
        query  += " AND status = ?"
        params.append(status)

    query += " ORDER BY upload_time DESC"

    conn = get_connection()
    try:
        cursor = conn.execute(query, params)
        rows   = cursor.fetchall()
    finally:
        conn.close()

    return [dict(row) for row in rows]


def save_approval(
    evidence_id: str,
    approved_by: str,
    block_index: int,
    timestamp: str,
    notes: str = "",
) -> None:
    """Record a blockchain approval event."""
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO approvals
                (evidence_id, approved_by, block_index, timestamp, notes)
            VALUES (?, ?, ?, ?, ?)
            """,
            (evidence_id, approved_by, block_index, timestamp, notes),
        )
        conn.commit()
    finally:
        conn.close()
