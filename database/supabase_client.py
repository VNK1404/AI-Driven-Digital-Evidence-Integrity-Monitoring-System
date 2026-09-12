"""
database/supabase_client.py
----------------------------
Supabase Database Client — AI-Driven Digital Evidence Integrity Monitoring System

Encapsulates CRUD operations for the 5 Supabase tables:
  1. evidence       (evidence_id PK)
  2. custody        (id identity PK)
  3. tamper_alerts  (id identity PK)
  4. approvals      (id identity PK)
  5. reports        (evidence_id PK)

Designed to work with either `supabase-py` or direct PostgREST HTTP calls over `requests`.
Graceful error handling: returns boolean/dict success flags without raising exceptions.
"""

import logging
from typing import Any
from config.settings import SUPABASE_URL, SUPABASE_KEY, is_supabase_enabled

logger = logging.getLogger(__name__)

_client_instance = None


def get_supabase_client():
    """
    Return singleton Supabase client or None if disabled/unconfigured.
    """
    global _client_instance
    if not is_supabase_enabled():
        return None

    if _client_instance is None:
        try:
            from supabase import create_client, Client
            _client_instance = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info("Supabase client initialized successfully (%s).", SUPABASE_URL)
        except ImportError:
            logger.warning("supabase-py package not installed. Using REST fallback.")
            _client_instance = "REST_FALLBACK"
        except Exception as exc:
            logger.error("Failed to initialize Supabase client: %s", exc)
            return None

    return _client_instance


# ── REST Helper fallback ──────────────────────────────────────────────────────

def _rest_request(table: str, method: str = "GET", data: dict | list | None = None, params: dict | None = None) -> Any:
    """Fallback REST helper using requests to communicate directly with PostgREST."""
    import requests
    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{table}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=representation",
    }
    try:
        if method == "POST":
            res = requests.post(url, headers=headers, json=data, timeout=10)
        elif method == "GET":
            res = requests.get(url, headers=headers, params=params, timeout=10)
        elif method == "PATCH":
            res = requests.patch(url, headers=headers, json=data, params=params, timeout=10)
        else:
            return None
        res.raise_for_status()
        return res.json()
    except Exception as exc:
        logger.error("Supabase REST request failed [%s %s]: %s", method, table, exc)
        return None


# ── Evidence Table Operations ─────────────────────────────────────────────────

def upload_file_to_supabase_storage(file_path: str, evidence_id: str, file_name: str) -> dict:
    """
    Upload physical evidence file binary (.jpg, .mp4, .pdf) to Supabase Storage bucket.

    Default bucket: 'evidence-files' (from SUPABASE_STORAGE_BUCKET)
    Storage Path: '<evidence_id>/<file_name>'

    Returns
    -------
    dict: {"storage_path": str, "storage_url": str}
    """
    client = get_supabase_client()
    if not client:
        return {"storage_path": "", "storage_url": ""}

    from config.settings import SUPABASE_STORAGE_BUCKET
    from pathlib import Path
    path_obj = Path(file_path)
    if not path_obj.exists():
        return {"storage_path": "", "storage_url": ""}

    storage_path = f"{evidence_id}/{file_name}"
    bucket = SUPABASE_STORAGE_BUCKET

    try:
        if client != "REST_FALLBACK":
            with path_obj.open("rb") as f:
                file_bytes = f.read()

            client.storage.from_(bucket).upload(
                path=storage_path,
                file=file_bytes,
                file_options={"upsert": "true"}
            )
            try:
                public_url = client.storage.from_(bucket).get_public_url(storage_path)
            except Exception:
                public_url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/public/{bucket}/{storage_path}"

            logger.info("[%s] Uploaded evidence file to Supabase Storage: %s", evidence_id, storage_path)
            return {"storage_path": storage_path, "storage_url": public_url}

        else:
            import requests
            url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/{bucket}/{storage_path}"
            headers = {
                "apikey": SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "x-upsert": "true",
            }
            with path_obj.open("rb") as f:
                res = requests.post(url, headers=headers, data=f, timeout=30)
            res.raise_for_status()
            public_url = f"{SUPABASE_URL.rstrip('/')}/storage/v1/object/public/{bucket}/{storage_path}"
            logger.info("[%s] Uploaded file via REST to Supabase Storage: %s", evidence_id, storage_path)
            return {"storage_path": storage_path, "storage_url": public_url}

    except Exception as exc:
        logger.warning("[%s] Supabase Storage upload warning (ensure bucket '%s' exists): %s", evidence_id, bucket, exc)
        return {"storage_path": storage_path, "storage_url": ""}


def save_evidence_supabase(evidence_dict: dict) -> bool:
    """Insert or upsert an evidence record into Supabase."""
    client = get_supabase_client()
    if not client:
        return False

    from datetime import datetime, timezone
    upload_time = evidence_dict.get("submitted_at") or evidence_dict.get("upload_time") or datetime.now(timezone.utc).isoformat()

    payload = {
        "evidence_id":  evidence_dict["evidence_id"],
        "file_name":    evidence_dict["file_name"],
        "file_path":    str(evidence_dict["file_path"]),
        "file_type":    evidence_dict["file_type"],
        "file_size":    evidence_dict["size_bytes"] if "size_bytes" in evidence_dict else evidence_dict.get("file_size", 0),
        "hash_value":   evidence_dict["sha256"] if "sha256" in evidence_dict else evidence_dict.get("hash_value", ""),
        "storage_path": evidence_dict.get("storage_path", ""),
        "storage_url":  evidence_dict.get("storage_url", ""),
        "uploaded_by":  evidence_dict.get("submitted_by", evidence_dict.get("uploaded_by", "system")),
        "upload_time":  upload_time,
        "status":       evidence_dict.get("status", "pending"),
    }

    try:
        if client == "REST_FALLBACK":
            res = _rest_request("evidence", method="POST", data=payload)
            return bool(res)
        res = client.table("evidence").upsert(payload).execute()
        return bool(res.data)
    except Exception as exc:
        logger.error("Supabase save_evidence failed: %s", exc)
        return False


def get_evidence_supabase(evidence_id: str) -> dict | None:
    """Fetch an evidence record by ID from Supabase."""
    client = get_supabase_client()
    if not client:
        return None

    try:
        if client == "REST_FALLBACK":
            res = _rest_request("evidence", method="GET", params={"evidence_id": f"eq.{evidence_id}"})
            return res[0] if res else None
        res = client.table("evidence").select("*").eq("evidence_id", evidence_id).execute()
        return res.data[0] if res.data else None
    except Exception as exc:
        logger.error("Supabase get_evidence failed for %s: %s", evidence_id, exc)
        return None


def update_evidence_status_supabase(evidence_id: str, status: str) -> bool:
    """Update evidence status in Supabase."""
    client = get_supabase_client()
    if not client:
        return False

    try:
        if client == "REST_FALLBACK":
            res = _rest_request("evidence", method="PATCH", data={"status": status}, params={"evidence_id": f"eq.{evidence_id}"})
            return bool(res)
        res = client.table("evidence").update({"status": status}).eq("evidence_id", evidence_id).execute()
        return bool(res.data)
    except Exception as exc:
        logger.error("Supabase update_evidence_status failed for %s: %s", evidence_id, exc)
        return False


# ── Custody Table Operations ───────────────────────────────────────────────────

def log_custody_event_supabase(evidence_id: str, action: str, user: str, notes: str, timestamp: str) -> bool:
    """Insert a chain-of-custody audit log into Supabase."""
    client = get_supabase_client()
    if not client:
        return False

    payload = {
        "evidence_id": evidence_id,
        "action": action,
        "user_name": user,
        "timestamp": timestamp,
        "notes": notes,
    }

    try:
        if client == "REST_FALLBACK":
            res = _rest_request("custody", method="POST", data=payload)
            return bool(res)
        res = client.table("custody").insert(payload).execute()
        return bool(res.data)
    except Exception as exc:
        logger.error("Supabase log_custody_event failed: %s", exc)
        return False


def get_custody_log_supabase(evidence_id: str) -> list[dict]:
    """Retrieve custody history for an evidence item from Supabase."""
    client = get_supabase_client()
    if not client:
        return []

    try:
        if client == "REST_FALLBACK":
            res = _rest_request("custody", method="GET", params={"evidence_id": f"eq.{evidence_id}", "order": "timestamp.asc"})
            return res or []
        res = client.table("custody").select("*").eq("evidence_id", evidence_id).order("timestamp", desc=False).execute()
        return res.data or []
    except Exception as exc:
        logger.error("Supabase get_custody_log failed for %s: %s", evidence_id, exc)
        return []


# ── Tamper Alerts Operations ─────────────────────────────────────────────────

def save_tamper_alert_supabase(alert_dict: dict) -> bool:
    """Insert a tamper alert into Supabase."""
    client = get_supabase_client()
    if not client:
        return False

    payload = {
        "evidence_id":  alert_dict["evidence_id"],
        "file_name":    alert_dict["file_name"],
        "file_type":    alert_dict["file_type"],
        "stored_hash":  alert_dict["stored_hash"],
        "current_hash": alert_dict["current_hash"],
        "detected_by":  alert_dict.get("detected_by", "system"),
        "timestamp":    alert_dict["timestamp"],
        "resolved":     alert_dict.get("resolved", False),
    }

    try:
        if client == "REST_FALLBACK":
            res = _rest_request("tamper_alerts", method="POST", data=payload)
            return bool(res)
        res = client.table("tamper_alerts").insert(payload).execute()
        return bool(res.data)
    except Exception as exc:
        logger.error("Supabase save_tamper_alert failed: %s", exc)
        return False


# ── Approvals Operations ──────────────────────────────────────────────────────

def save_approval_supabase(evidence_id: str, approved_by: str, block_index: int, timestamp: str, notes: str) -> bool:
    """Insert an officer approval record into Supabase."""
    client = get_supabase_client()
    if not client:
        return False

    payload = {
        "evidence_id": evidence_id,
        "approved_by": approved_by,
        "block_index": block_index,
        "timestamp": timestamp,
        "notes": notes,
    }

    try:
        if client == "REST_FALLBACK":
            res = _rest_request("approvals", method="POST", data=payload)
            return bool(res)
        res = client.table("approvals").insert(payload).execute()
        return bool(res.data)
    except Exception as exc:
        logger.error("Supabase save_approval failed: %s", exc)
        return False


# ── Forensic Reports Operations ───────────────────────────────────────────────

def save_report_supabase(report_dict: dict) -> bool:
    """Insert or upsert a full JSON ForensicReport into Supabase."""
    client = get_supabase_client()
    if not client:
        return False

    payload = {
        "evidence_id":     report_dict["evidence_id"],
        "overall_status":  report_dict["overall_status"],
        "file_metadata":   report_dict["file"],
        "blockchain_res":  report_dict.get("blockchain"),
        "metadata_res":    report_dict.get("metadata"),
        "forgery_res":     report_dict.get("image_forgery"),
        "deepfake_res":    report_dict.get("deepfake"),
        "fake_news_res":   report_dict.get("fake_news"),
        "created_at":      report_dict.get("analysis_timestamp"),
    }

    try:
        if client == "REST_FALLBACK":
            res = _rest_request("reports", method="POST", data=payload)
            return bool(res)
        res = client.table("reports").upsert(payload).execute()
        return bool(res.data)
    except Exception as exc:
        logger.error("Supabase save_report failed: %s", exc)
        return False


def get_report_supabase(evidence_id: str) -> dict | None:
    """Fetch a saved forensic report by ID from Supabase."""
    client = get_supabase_client()
    if not client:
        return None

    try:
        if client == "REST_FALLBACK":
            res = _rest_request("reports", method="GET", params={"evidence_id": f"eq.{evidence_id}"})
            if not res: return None
            r = res[0]
        else:
            res = client.table("reports").select("*").eq("evidence_id", evidence_id).execute()
            if not res.data: return None
            r = res.data[0]

        # Reconstruct standard ForensicReport dictionary
        return {
            "evidence_id":        r["evidence_id"],
            "file":               r["file_metadata"],
            "blockchain":         r.get("blockchain_res"),
            "metadata":           r.get("metadata_res"),
            "image_forgery":      r.get("forgery_res"),
            "deepfake":           r.get("deepfake_res"),
            "fake_news":          r.get("fake_news_res"),
            "overall_status":     r["overall_status"],
            "analysis_timestamp": r.get("created_at"),
        }
    except Exception as exc:
        logger.error("Supabase get_report failed for %s: %s", evidence_id, exc)
        return None
