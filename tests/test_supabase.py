"""
tests/test_supabase.py
-----------------------
Unit tests for database/supabase_client.py, database/db_service.py,
and Supabase schema script verification.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pytest
from config.settings import is_supabase_enabled
from database.supabase_client import get_supabase_client
from database import db_service


class TestSupabaseConfiguration:
    def test_schema_file_exists(self):
        sql_path = _PROJECT_ROOT / "supabase_schema.sql"
        assert sql_path.exists()
        content = sql_path.read_text(encoding="utf-8")
        assert "CREATE TABLE IF NOT EXISTS public.evidence" in content
        assert "CREATE TABLE IF NOT EXISTS public.custody" in content
        assert "CREATE TABLE IF NOT EXISTS public.tamper_alerts" in content
        assert "CREATE TABLE IF NOT EXISTS public.approvals" in content
        assert "CREATE TABLE IF NOT EXISTS public.reports" in content

    def test_graceful_fallback_when_unconfigured(self, monkeypatch):
        # Disable Supabase explicitly for this test
        monkeypatch.setattr("config.settings.SUPABASE_URL", "")
        monkeypatch.setattr("config.settings.SUPABASE_KEY", "")
        assert is_supabase_enabled() is False
        assert get_supabase_client() is None

    def test_db_service_fallback_operations(self, tmp_path):
        """Test that db_service functions complete via SQLite even without Supabase credentials."""
        eid = "EV-SUPA-TEST-001"
        dummy_file = tmp_path / "test.jpg"
        dummy_file.write_bytes(b"data")

        # Save record
        db_service.save_evidence_record(
            evidence_id=eid,
            file_name="test.jpg",
            file_path=str(dummy_file),
            file_type="image",
            file_size=4,
            hash_value="abc123hash",
            uploaded_by="tester",
            upload_time="2026-08-11T20:00:00Z",
        )

        # Retrieve record
        rec = db_service.get_evidence_record(eid)
        assert rec is not None
        assert rec["evidence_id"] == eid

        # Log custody
        db_service.log_custody_event_record(eid, "upload", "tester", "Test upload")
        custody = db_service.get_custody_log_records(eid)
        assert isinstance(custody, list)

        # Update status
        db_service.update_evidence_status_record(eid, "verified")
