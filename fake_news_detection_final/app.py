"""
app.py — Flask Web Interface for Fake News Detection Module
AI-Driven Digital Evidence Integrity Monitoring System
"""

import os
import sys
import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_news_module.analytics.dashboard_api import analytics_bp
from fake_news_module.reporting.report_generator import (
    build_report_payload,
    cleanup_old_reports,
    generate_pdf_report,
)
from fake_news_module.config import SUPPORTED_IMAGE_EXTENSIONS, SUPPORTED_PDF_EXTENSIONS

# ──────────────────────────────────────────────
# App Configuration
# ──────────────────────────────────────────────
UPLOAD_FOLDER = Path(__file__).parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
REPORTS_FOLDER = Path(__file__).parent / "reports"
REPORTS_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_PDF_EXTENSIONS
MAX_CONTENT_LENGTH = 20 * 1024 * 1024   # 20 MB
REPORT_RETENTION_HOURS = 24
REPORT_REGISTRY_LIMIT = 50
REPORT_REGISTRY: dict[str, dict[str, object]] = {}

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(
    app,
    resources={
        r"/*": {
            "origins": [
                "http://localhost:3000",
                "http://127.0.0.1:3000",
            ]
        }
    },
    expose_headers=["X-Analysis-Id"],
)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.secret_key = os.environ.get("FLASK_SECRET")
if not app.secret_key:
    raise RuntimeError(
        "FLASK_SECRET environment variable not set."
    )
app.register_blueprint(analytics_bp)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fake_news_pipeline(input_value: str):
    from fake_news_module.core.pipeline import fake_news_pipeline as _pipeline

    return _pipeline(input_value)


def analyze_text_pipeline(text: str):
    from fake_news_module.core.pipeline import analyze_text_pipeline as _pipeline

    return _pipeline(text)


def allowed_file(filename: str) -> bool:
    suffix = Path(filename).suffix.lower()
    return suffix in ALLOWED_EXTENSIONS


def _store_reportable_result(result: dict) -> str:
    analysis_id = uuid.uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    stored_result = dict(result)
    stored_result["analysis_timestamp"] = created_at
    REPORT_REGISTRY[analysis_id] = {
        "result": stored_result,
        "created_at": created_at,
    }
    if len(REPORT_REGISTRY) > REPORT_REGISTRY_LIMIT:
        oldest = next(iter(REPORT_REGISTRY))
        REPORT_REGISTRY.pop(oldest, None)
    return analysis_id


def _resolve_report_source() -> tuple[dict, str | None]:
    payload: dict | None = None
    analysis_id = request.args.get("analysis_id")

    request_json = request.get_json(silent=True)
    if isinstance(request_json, dict):
        analysis_id = analysis_id or str(request_json.get("analysis_id") or "")
        if isinstance(request_json.get("result"), dict):
            payload = request_json["result"]
        elif "final_decision" in request_json:
            payload = request_json

    if payload is None and request.form:
        form_analysis_id = request.form.get("analysis_id")
        if form_analysis_id:
            analysis_id = analysis_id or form_analysis_id
        raw_result = request.form.get("result")
        if raw_result:
            try:
                parsed = json.loads(raw_result)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                payload = parsed

    if payload is None and analysis_id and analysis_id in REPORT_REGISTRY:
        stored = REPORT_REGISTRY.get(analysis_id, {}).get("result")
        if isinstance(stored, dict):
            payload = stored

    if payload is None:
        raise ValueError("No report data supplied. Provide analysis_id or result JSON.")

    return payload, analysis_id


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "fake-news-detection",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }), 200


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Accepts a text claim or multipart file upload, runs the detection pipeline,
    and returns a JSON result.
    """
    claim_text = (request.form.get("text") or "").strip()
    if claim_text:
        try:
            result = analyze_text_pipeline(claim_text)
            analysis_id = _store_reportable_result(result)
            response = jsonify(result)
            response.headers["X-Analysis-Id"] = analysis_id
            return response, 200

        except ValueError as exc:
            logger.warning("Validation error: %s", exc)
            return jsonify({"error": str(exc)}), 422

        except RuntimeError as exc:
            logger.error("Runtime error: %s", exc)
            return jsonify({"error": str(exc)}), 500

        except Exception as exc:                      # noqa: BLE001
            logger.exception("Unexpected error: %s", exc)
            return jsonify({"error": "An unexpected error occurred."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: jpg, jpeg, png, bmp, pdf"
        }), 415

    # Save with a unique name to avoid collisions
    original_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    save_path = UPLOAD_FOLDER / unique_name

    try:
        file.save(str(save_path))
        logger.info("File saved: %s", save_path)

        result = fake_news_pipeline(str(save_path))
        result["filename"] = original_name
        analysis_id = _store_reportable_result(result)
        response = jsonify(result)
        response.headers["X-Analysis-Id"] = analysis_id
        return response, 200

    except FileNotFoundError as exc:
        logger.error("File not found: %s", exc)
        return jsonify({"error": str(exc)}), 404

    except ValueError as exc:
        logger.warning("Validation error: %s", exc)
        return jsonify({"error": str(exc)}), 422

    except RuntimeError as exc:
        logger.error("Runtime error: %s", exc)
        return jsonify({"error": str(exc)}), 500

    except Exception as exc:                      # noqa: BLE001
        logger.exception("Unexpected error: %s", exc)
        return jsonify({"error": "An unexpected error occurred."}), 500

    finally:
        # Clean up uploaded file after processing
        if save_path.exists():
            save_path.unlink(missing_ok=True)


@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "File too large. Maximum size is 20 MB."}), 413


@app.route("/download_report", methods=["GET", "POST"])
def download_report():
    try:
        result, analysis_id = _resolve_report_source()
        cleanup_old_reports(REPORTS_FOLDER, max_age_hours=REPORT_RETENTION_HOURS)
        report_payload = build_report_payload(result, analysis_id=analysis_id)
        report_id = analysis_id or uuid.uuid4().hex
        report_path = REPORTS_FOLDER / f"{report_id}.pdf"
        generate_pdf_report(report_payload, report_path)

        response = send_file(
            report_path,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"fake_news_report_{report_id}.pdf",
        )

        @response.call_on_close
        def _cleanup_report_file() -> None:
            try:
                report_path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Temporary report cleanup skipped for %s", report_path)

        return response

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        logger.exception("Report generation failed: %s", exc)
        return jsonify({"error": "Failed to generate PDF report."}), 500


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Fake News Detection - Web Interface ===")
    print("=== Open: http://127.0.0.1:5000         ===")
    print("=========================================\n")
    app.run(debug=True, host="127.0.0.1", port=5000)
