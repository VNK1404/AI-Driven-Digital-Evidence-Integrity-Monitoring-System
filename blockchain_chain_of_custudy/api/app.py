"""
app.py
------
Flask REST API — Evidence Integrity & Blockchain Ledger System

Supports: Images, Videos, Audio, Documents

WORKFLOW:
  1. POST /upload    → Register any file (image/video/audio/doc), store hash
  2. POST /verify    → Check if file is tampered (alert saved if tampered)
  3. POST /approve   → Officer manually approves verified file → blockchain
  4. GET  /blockchain → View full approved evidence ledger
  5. GET  /alerts     → View all tamper alerts
  6. GET  /evidence   → List all files (filter by type/status)
  7. GET  /custody/<id> → Full audit trail for one file

PORT: 5001
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from flask import Flask, request, jsonify

from database.db import (
    init_db, save_evidence, get_evidence, get_all_evidence,
    update_evidence_status, save_approval,
)
from integrity.hashing   import generate_hash, get_file_metadata
from integrity.blockchain import evidence_blockchain
from integrity.verify    import verify_integrity, STATUS_VERIFIED, STATUS_TAMPERED
from integrity.custody   import (
    log_custody_event, save_tamper_alert, get_all_alerts, get_custody_log,
    ACTION_UPLOAD, ACTION_VERIFY, ACTION_APPROVE, ACTION_ACCESS,
)

app = Flask(__name__)
init_db()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _error(message: str, code: int = 400):
    return jsonify({"success": False, "error": message}), code

def _ok(data: dict, code: int = 200):
    return jsonify({"success": True, **data}), code


# ── Health Check ──────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Service health check."""
    return _ok({
        "module":    "Evidence Integrity & Blockchain Ledger",
        "version":   "2.0",
        "timestamp": datetime.utcnow().isoformat(),
        "supported_types": {
            "image":    ["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
            "video":    ["mp4", "avi", "mkv", "mov"],
            "audio":    ["mp3", "wav", "aac", "flac"],
            "document": ["pdf", "docx", "txt", "xlsx"],
        },
    })


# ── Upload ────────────────────────────────────────────────────────────────────
@app.route("/upload", methods=["POST"])
def upload():
    """
    Register a new evidence file (any supported type).

    Request JSON:
        {
            "file_path":   "C:/evidence/video.mp4",
            "evidence_id": "EV001",
            "user":        "officer_parth"   (optional)
        }

    Response:
        evidence_id, file_name, file_type, hash, file_size, status
    """
    body = request.get_json(silent=True)
    if not body:
        return _error("Request body must be valid JSON.")

    file_path   = body.get("file_path",   "").strip()
    evidence_id = body.get("evidence_id", "").strip()
    user        = body.get("user", "system").strip()

    if not file_path:
        return _error("'file_path' is required.")
    if not evidence_id:
        return _error("'evidence_id' is required.")

    # Prevent duplicate IDs
    if get_evidence(evidence_id):
        return _error(f"Evidence ID '{evidence_id}' already exists.", code=409)

    # Hash the file and get metadata
    try:
        hash_value = generate_hash(file_path)
        metadata   = get_file_metadata(file_path)
    except FileNotFoundError as e:
        return _error(str(e), code=404)
    except ValueError as e:
        return _error(str(e), code=415)   # Unsupported Media Type
    except Exception as e:
        return _error(f"Hashing failed: {str(e)}", code=500)

    # Save to database (status = pending)
    upload_time = datetime.utcnow().isoformat()
    try:
        save_evidence(
            evidence_id = evidence_id,
            file_name   = metadata["file_name"],
            file_path   = file_path,
            file_type   = metadata["file_type"],
            file_size   = metadata["file_size"],
            hash_value  = hash_value,
            uploaded_by = user,
            upload_time = upload_time,
        )
    except Exception as e:
        return _error(f"Database error: {str(e)}", code=500)

    # Log custody
    log_custody_event(
        evidence_id = evidence_id,
        action      = ACTION_UPLOAD,
        user        = user,
        notes       = (
            f"Uploaded {metadata['file_type']} file: {metadata['file_name']}. "
            f"Size: {metadata['file_size']:,} bytes. "
            f"Hash: {hash_value[:20]}..."
        ),
    )

    return _ok({
        "evidence_id": evidence_id,
        "file_name":   metadata["file_name"],
        "file_type":   metadata["file_type"],
        "file_size":   metadata["file_size"],
        "hash":        hash_value,
        "upload_time": upload_time,
        "status":      "pending",
        "next_step":   "Run POST /verify to check integrity before approving.",
    }, code=201)


# ── Verify ────────────────────────────────────────────────────────────────────
@app.route("/verify", methods=["POST"])
def verify():
    """
    Verify file integrity. Saves tamper alert to DB if tampered.

    Request JSON:
        {
            "file_path":   "C:/evidence/video.mp4",
            "evidence_id": "EV001",
            "user":        "officer_parth"
        }

    Response:
        integrity_status (VERIFIED | TAMPERED | ERROR),
        alert saved to DB if tampered
    """
    body = request.get_json(silent=True)
    if not body:
        return _error("Request body must be valid JSON.")

    file_path   = body.get("file_path",   "").strip()
    evidence_id = body.get("evidence_id", "").strip()
    user        = body.get("user", "system").strip()

    if not file_path:
        return _error("'file_path' is required.")
    if not evidence_id:
        return _error("'evidence_id' is required.")

    # Fetch stored record
    record = get_evidence(evidence_id)
    if not record:
        return _error(f"No evidence found for ID '{evidence_id}'.", code=404)

    # Run integrity check
    result = verify_integrity(file_path, record["hash_value"])

    alert_saved = False
    alert_id    = None

    if result["status"] == STATUS_TAMPERED:
        # Save tamper alert to database
        update_evidence_status(evidence_id, "tampered")
        alert = save_tamper_alert(
            evidence_id  = evidence_id,
            file_name    = record["file_name"],
            file_type    = record["file_type"],
            stored_hash  = record["hash_value"],
            current_hash = result["current_hash"],
            detected_by  = user,
        )
        alert_saved = True
        alert_id    = alert["alert_id"]

    elif result["status"] == STATUS_VERIFIED:
        # Update status to verified (ready for approval)
        update_evidence_status(evidence_id, "verified")

    # Log custody event
    log_custody_event(
        evidence_id = evidence_id,
        action      = ACTION_VERIFY,
        user        = user,
        notes       = f"Integrity check: {result['status']}.",
    )

    response_data = {
        "evidence_id":      evidence_id,
        "file_name":        record["file_name"],
        "file_type":        record["file_type"],
        "integrity_status": result["status"],
        "current_hash":     result["current_hash"],
        "stored_hash":      result["stored_hash"],
        "message":          result["message"],
        "timestamp":        datetime.utcnow().isoformat(),
        "alert_saved":      alert_saved,
        "alert_id":         alert_id,
    }

    if result["status"] == STATUS_VERIFIED:
        response_data["next_step"] = (
            "File is clean. Run POST /approve to add to blockchain ledger."
        )
    elif result["status"] == STATUS_TAMPERED:
        response_data["next_step"] = (
            "Tampering alert saved. This file cannot be approved for blockchain."
        )

    return _ok(response_data)


# ── Approve → Blockchain ──────────────────────────────────────────────────────
@app.route("/approve", methods=["POST"])
def approve():
    """
    Officer manually approves a VERIFIED file → adds to blockchain.

    Only files with status='verified' can be approved.
    Tampered or pending files are rejected.

    Request JSON:
        {
            "evidence_id": "EV001",
            "approved_by": "officer_parth",
            "notes":       "Reviewed and confirmed authentic"  (optional)
        }

    Response:
        block_index, block_hash, evidence_id
    """
    body = request.get_json(silent=True)
    if not body:
        return _error("Request body must be valid JSON.")

    evidence_id = body.get("evidence_id", "").strip()
    approved_by = body.get("approved_by", "").strip()
    notes       = body.get("notes", "").strip()

    if not evidence_id:
        return _error("'evidence_id' is required.")
    if not approved_by:
        return _error("'approved_by' is required.")

    # Fetch record
    record = get_evidence(evidence_id)
    if not record:
        return _error(f"No evidence found for ID '{evidence_id}'.", code=404)

    # Only verified files can be approved
    if record["status"] == "tampered":
        return _error(
            f"Evidence '{evidence_id}' has been flagged as TAMPERED. "
            "It cannot be approved for the blockchain.",
            code=403,
        )
    if record["status"] == "pending":
        return _error(
            f"Evidence '{evidence_id}' has not been verified yet. "
            "Run POST /verify first.",
            code=403,
        )
    if record["status"] == "approved":
        return _error(
            f"Evidence '{evidence_id}' is already on the blockchain.",
            code=409,
        )

    # Add to blockchain
    timestamp = datetime.utcnow().isoformat()
    block = evidence_blockchain.add_approved_block(
        evidence_data = {
            "evidence_id": evidence_id,
            "file_name":   record["file_name"],
            "file_type":   record["file_type"],
            "hash_value":  record["hash_value"],
            "file_size":   record["file_size"],
        },
        approved_by = approved_by,
    )

    # Update DB status → approved
    update_evidence_status(evidence_id, "approved")

    # Save approval record
    save_approval(
        evidence_id = evidence_id,
        approved_by = approved_by,
        block_index = block.index,
        timestamp   = timestamp,
        notes       = notes,
    )

    # Log custody
    log_custody_event(
        evidence_id = evidence_id,
        action      = ACTION_APPROVE,
        user        = approved_by,
        notes       = (
            f"Approved for blockchain. Block #{block.index}. "
            f"Block hash: {block.hash[:20]}... Notes: {notes}"
        ),
    )

    return _ok({
        "evidence_id": evidence_id,
        "file_name":   record["file_name"],
        "file_type":   record["file_type"],
        "approved_by": approved_by,
        "block_index": block.index,
        "block_hash":  block.hash,
        "timestamp":   timestamp,
        "message":     (
            f"Evidence '{evidence_id}' successfully added to the blockchain "
            f"at block #{block.index}."
        ),
    }, code=201)


# ── Blockchain ────────────────────────────────────────────────────────────────
@app.route("/blockchain", methods=["GET"])
def get_blockchain():
    """
    Return the full blockchain ledger of approved evidence.

    Response:
        chain_length, chain_valid, blockchain (list of blocks)
    """
    chain = evidence_blockchain.get_chain()
    return _ok({
        "chain_length": len(chain),
        "chain_valid":  evidence_blockchain.is_chain_valid(),
        "blockchain":   chain,
    })


@app.route("/blockchain/<evidence_id>", methods=["GET"])
def find_in_blockchain(evidence_id: str):
    """Find a specific evidence block by evidence_id."""
    block = evidence_blockchain.find_block(evidence_id)
    if not block:
        return _error(
            f"Evidence '{evidence_id}' not found in blockchain. "
            "It may not have been approved yet.",
            code=404,
        )
    return _ok({"block": block})


# ── Alerts ────────────────────────────────────────────────────────────────────
@app.route("/alerts", methods=["GET"])
def get_alerts():
    """
    Return all tamper alerts saved in the database.

    Query params:
        ?resolved=true  → Show resolved alerts (default: unresolved)
    """
    resolved = request.args.get("resolved", "false").lower() == "true"
    alerts   = get_all_alerts(resolved=resolved)
    return _ok({
        "alert_count": len(alerts),
        "resolved":    resolved,
        "alerts":      alerts,
    })


# ── Evidence List ─────────────────────────────────────────────────────────────
@app.route("/evidence", methods=["GET"])
def list_evidence():
    """
    List all evidence records with optional filters.

    Query params:
        ?type=image|video|audio|document
        ?status=pending|verified|approved|tampered
    """
    file_type = request.args.get("type",   None)
    status    = request.args.get("status", None)
    records   = get_all_evidence(file_type=file_type, status=status)
    return _ok({
        "count":    len(records),
        "filters":  {"type": file_type, "status": status},
        "evidence": records,
    })


@app.route("/evidence/<evidence_id>", methods=["GET"])
def get_evidence_detail(evidence_id: str):
    """Get a single evidence record and log access."""
    record = get_evidence(evidence_id)
    if not record:
        return _error(f"Evidence '{evidence_id}' not found.", code=404)

    log_custody_event(
        evidence_id = evidence_id,
        action      = ACTION_ACCESS,
        user        = request.args.get("user", "system"),
        notes       = "Record accessed via API.",
    )
    return _ok({"evidence": record})


# ── Custody Log ───────────────────────────────────────────────────────────────
@app.route("/custody/<evidence_id>", methods=["GET"])
def custody_log(evidence_id: str):
    """Return the full chain-of-custody audit trail for one evidence item."""
    log = get_custody_log(evidence_id)
    return _ok({
        "evidence_id": evidence_id,
        "event_count": len(log),
        "custody_log": log,
    })


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 62)
    print("  AI-Based Evidence Integrity Monitoring System  v2.0")
    print("  Module: Cryptographic Integrity & Blockchain Ledger")
    print("=" * 62)
    print(f"  API Base  : http://127.0.0.1:5001")
    print(f"  Supports  : Images | Videos | Audio | Documents")
    print(f"  Blockchain: Manual approval only (verified files)")
    print("=" * 62)
    app.run(host="0.0.0.0", port=5001, debug=True)
