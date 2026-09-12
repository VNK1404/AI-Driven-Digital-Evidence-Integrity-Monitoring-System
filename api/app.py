"""
api/app.py
-----------
Unified Flask REST API — AI-Driven Digital Evidence Integrity Monitoring System

Port: 8080  (all other existing servers use 5000, 5001, 8000)

Endpoints
---------
GET  /health                    → system health + chain status
POST /analyze                   → upload file for full analysis
GET  /report/<evidence_id>      → retrieve saved report JSON
GET  /reports                   → list all report evidence IDs
GET  /custody/<evidence_id>     → get custody log
POST /approve/<evidence_id>     → officer blockchain approval
GET  /chain                     → full blockchain ledger

Design
------
- Accepts multipart/form-data file uploads.
- Saves the upload to a temp file under reports/uploads/.
- Calls analyze_evidence() from the orchestrator.
- Returns JSON.  ALL errors are JSON — never HTML 500 pages.
- File cleanup: uploaded temp files are deleted after analysis.
"""

import os
import json
import logging
import tempfile
import traceback
from pathlib import Path

from flask import Flask, jsonify, request, abort, render_template
from flask_cors import CORS

from config.settings import (
    API_HOST, API_PORT, API_DEBUG,
    REPORTS_DIR, configure_logging,
)
from orchestration.analysis_orchestrator import (
    analyze_evidence, load_report, list_reports,
)
from orchestration.evidence_router import route_evidence

configure_logging("DEBUG" if API_DEBUG else "INFO")
logger = logging.getLogger(__name__)

# ── Upload staging directory ───────────────────────────────────────────────────
UPLOAD_DIR = REPORTS_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
CORS(app)   # Allow cross-origin requests (useful for frontend integration)

app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024   # 2 GB upload limit


# ── Error handlers ─────────────────────────────────────────────────────────────

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request", "message": str(e)}), 400


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found", "message": str(e)}), 404


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "File too large", "message": "Max upload size is 2 GB"}), 413


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error", "message": str(e)}), 500


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
@app.route("/ui", methods=["GET"])
def index():
    """Render testing UI workbench."""
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    """
    GET /health
    System health check.  Always returns 200.
    """
    try:
        from integrations.blockchain.adapter import get_chain_status
        chain = get_chain_status()
    except Exception:
        chain = {"error": "blockchain not initialised"}

    return jsonify({
        "status": "healthy",
        "api_version": "1.0.0",
        "system": "AI-Driven Digital Evidence Integrity Monitoring System",
        "blockchain": chain,
        "report_count": len(list_reports()),
    }), 200


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    POST /analyze
    Upload an evidence file for full forensic analysis.

    Form fields
    -----------
    file         : required — the evidence file (multipart)
    submitted_by : optional — user identifier (default: 'api_user')

    Returns
    -------
    JSON ForensicReport (same structure as reports/<id>.json)
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided", "message": "Include 'file' in multipart form data"}), 400

    uploaded = request.files["file"]
    if not uploaded.filename:
        return jsonify({"error": "Empty filename"}), 400

    submitted_by = request.form.get("submitted_by", "api_user")

    # Save to temp file preserving original extension
    suffix = Path(uploaded.filename).suffix.lower()
    tmp_path = None
    try:
        # Save upload
        fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=UPLOAD_DIR)
        with os.fdopen(fd, "wb") as f:
            uploaded.save(f)

        logger.info(
            "Received upload: '%s' → %s (submitted_by=%s)",
            uploaded.filename, tmp_path, submitted_by,
        )

        # Check routing before analysis
        decision = route_evidence(tmp_path)
        if not decision.is_supported:
            return jsonify({
                "error": "Unsupported file type",
                "message": decision.reject_reason,
            }), 415

        # Full analysis
        report = analyze_evidence(tmp_path, submitted_by=submitted_by, original_filename=uploaded.filename)
        return jsonify(report.to_dict()), 200

    except FileNotFoundError as exc:
        return jsonify({"error": "File not found", "message": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": "Invalid input", "message": str(exc)}), 400
    except Exception as exc:
        logger.error("analyze endpoint error: %s\n%s", exc, traceback.format_exc())
        return jsonify({"error": "Analysis failed", "message": str(exc)}), 500
    finally:
        # Clean up temp upload file
        if tmp_path and Path(tmp_path).exists():
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass


@app.route("/analyze/path", methods=["POST"])
def analyze_by_path():
    """
    POST /analyze/path
    Analyze a file that already exists on the server filesystem.

    JSON body
    ---------
    { "file_path": "/absolute/path/to/file", "submitted_by": "user" }

    Useful for batch processing and CLI-to-API bridging.
    """
    body = request.get_json(silent=True) or {}
    file_path = body.get("file_path")
    submitted_by = body.get("submitted_by", "api_user")

    if not file_path:
        return jsonify({"error": "Missing 'file_path' in request body"}), 400

    try:
        report = analyze_evidence(file_path, submitted_by=submitted_by)
        return jsonify(report.to_dict()), 200
    except FileNotFoundError as exc:
        return jsonify({"error": "File not found", "message": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"error": "Invalid input", "message": str(exc)}), 400
    except Exception as exc:
        logger.error("analyze/path endpoint error: %s", exc, exc_info=True)
        return jsonify({"error": "Analysis failed", "message": str(exc)}), 500


@app.route("/report/<evidence_id>", methods=["GET"])
def get_report(evidence_id: str):
    """
    GET /report/<evidence_id>
    Retrieve a previously saved ForensicReport.
    """
    report = load_report(evidence_id)
    if report is None:
        return jsonify({
            "error": "Report not found",
            "evidence_id": evidence_id,
        }), 404
    return jsonify(report), 200


@app.route("/reports", methods=["GET"])
def get_reports():
    """
    GET /reports
    List all evidence IDs that have saved reports.
    """
    ids = list_reports()
    return jsonify({"count": len(ids), "evidence_ids": ids}), 200


@app.route("/custody/<evidence_id>", methods=["GET"])
def get_custody(evidence_id: str):
    """
    GET /custody/<evidence_id>
    Return the full chain-of-custody log for one evidence item.
    """
    try:
        from integrations.blockchain.adapter import get_custody_log
        log = get_custody_log(evidence_id)
        return jsonify({"evidence_id": evidence_id, "custody_log": log}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/approve/<evidence_id>", methods=["POST"])
def approve(evidence_id: str):
    """
    POST /approve/<evidence_id>
    Officer approves evidence — adds it to the blockchain ledger.

    JSON body (optional)
    --------------------
    { "approved_by": "officer_name", "notes": "Approval notes" }
    """
    body = request.get_json(silent=True) or {}
    approved_by = body.get("approved_by", "officer")
    notes = body.get("notes", "")

    try:
        from integrations.blockchain.adapter import approve_evidence
        result = approve_evidence(
            evidence_id=evidence_id,
            approved_by=approved_by,
            notes=notes,
        )
        return jsonify(result.to_dict()), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/chain", methods=["GET"])
def get_chain():
    """
    GET /chain
    Return the full in-memory blockchain ledger.
    """
    try:
        from integrations.blockchain.adapter import get_chain_status
        return jsonify(get_chain_status()), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/verify/<evidence_id>", methods=["POST"])
def verify(evidence_id: str):
    """
    POST /verify/<evidence_id>
    Verify the integrity of a registered evidence file.

    JSON body
    ---------
    { "file_path": "/absolute/path/to/file" }
    """
    body = request.get_json(silent=True) or {}
    file_path = body.get("file_path")

    if not file_path:
        return jsonify({"error": "Missing 'file_path' in request body"}), 400

    try:
        from integrations.blockchain.adapter import verify_integrity
        result = verify_integrity(evidence_id=evidence_id, file_path=file_path)
        return jsonify(result.to_dict()), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Entry point ────────────────────────────────────────────────────────────────

def create_app() -> Flask:
    """Application factory — returns the configured Flask app."""
    return app


if __name__ == "__main__":
    logger.info(
        "Starting AI-Driven Forensic System API on %s:%s",
        API_HOST, API_PORT,
    )
    app.run(host=API_HOST, port=API_PORT, debug=API_DEBUG)
