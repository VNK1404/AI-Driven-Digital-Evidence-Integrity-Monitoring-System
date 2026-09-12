"""
tests/test_orchestrator.py
----------------------------
Unit tests for orchestration/analysis_orchestrator.py

Tests the orchestrator without starting the API server.
Mocks adapters where possible to isolate orchestration logic.
"""

import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pytest
from schemas.results import ModuleResult


class TestOrchestratorFileValidation:
    """Test pre-flight checks (no model loading)."""

    def test_missing_file_raises(self, tmp_path):
        from orchestration.analysis_orchestrator import analyze_evidence
        with pytest.raises(FileNotFoundError):
            analyze_evidence(str(tmp_path / "nonexistent.jpg"))

    def test_unsupported_file_raises(self, tmp_path):
        from orchestration.analysis_orchestrator import analyze_evidence
        f = tmp_path / "archive.zip"
        f.write_bytes(b"zip")
        with pytest.raises(ValueError, match="nsupported"):
            analyze_evidence(str(f))

    def test_audio_file_raises(self, tmp_path):
        from orchestration.analysis_orchestrator import analyze_evidence
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"mp3")
        with pytest.raises(ValueError):
            analyze_evidence(str(f))


class TestOrchestratorWithMocks:
    """
    Test orchestrator flow by mocking all adapters.
    Verifies that:
    - evidence_id is consistent across all module results
    - module failures don't crash others
    - report is saved to disk
    """

    def _make_success(self, eid: str, atype: str) -> ModuleResult:
        return ModuleResult(
            evidence_id=eid,
            analysis_type=atype,
            status="success",
            result={"test_key": "test_value"},
        )

    @patch("integrations.blockchain.adapter.register_evidence")
    @patch("integrations.metadata.adapter.analyze_metadata")
    @patch("integrations.image_forgery.adapter.analyze_image_forgery")
    @patch("integrations.fake_news.adapter.is_news_content_applicable", return_value=False)
    def test_image_flow_all_success(
        self,
        mock_fn_check,
        mock_forgery,
        mock_metadata,
        mock_blockchain,
        tmp_path,
    ):
        from orchestration.analysis_orchestrator import analyze_evidence

        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)

        # The mocks need to return ModuleResult with the correct evidence_id
        def make_result(atype):
            def _fn(*args, **kwargs):
                eid = kwargs.get("evidence_id") or args[0]
                return self._make_success(eid, atype)
            return _fn

        mock_blockchain.side_effect = make_result("blockchain")
        mock_metadata.side_effect = make_result("metadata_forensics")
        mock_forgery.side_effect = make_result("image_forgery")

        report = analyze_evidence(str(img))
        rd = report.to_dict()

        assert rd["overall_status"] == "completed"
        assert rd["blockchain"]["status"] == "success"
        assert rd["metadata"]["status"] == "success"
        assert rd["image_forgery"]["status"] == "success"
        assert rd["deepfake"] is None   # not scheduled for images
        # fake_news skipped because mock returns False
        assert rd["fake_news"]["status"] == "skipped"

        # Verify report saved to disk
        eid = rd["evidence_id"]
        report_file = _PROJECT_ROOT / "reports" / f"{eid}.json"
        assert report_file.exists()
        report_file.unlink()   # cleanup

    @patch("integrations.blockchain.adapter.register_evidence")
    @patch("integrations.metadata.adapter.analyze_metadata")
    @patch("integrations.image_forgery.adapter.analyze_image_forgery")
    @patch("integrations.fake_news.adapter.is_news_content_applicable", return_value=False)
    def test_one_module_failure_does_not_crash_others(
        self,
        mock_fn_check,
        mock_forgery,
        mock_metadata,
        mock_blockchain,
        tmp_path,
    ):
        from orchestration.analysis_orchestrator import analyze_evidence

        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG" + b"\x00" * 100)

        def blockchain_ok(*args, **kwargs):
            eid = kwargs.get("evidence_id") or args[0]
            return self._make_success(eid, "blockchain")

        def metadata_fail(*args, **kwargs):
            raise RuntimeError("Simulated metadata crash")

        def forgery_ok(*args, **kwargs):
            eid = kwargs.get("evidence_id") or args[0]
            return self._make_success(eid, "image_forgery")

        mock_blockchain.side_effect = blockchain_ok
        mock_metadata.side_effect = metadata_fail
        mock_forgery.side_effect = forgery_ok

        # Must NOT raise
        report = analyze_evidence(str(img))
        rd = report.to_dict()

        assert rd["blockchain"]["status"] == "success"
        assert rd["metadata"]["status"] == "failed"          # caught gracefully
        assert rd["image_forgery"]["status"] == "success"
        assert rd["overall_status"] == "partial"

        # Cleanup
        eid = rd["evidence_id"]
        report_file = _PROJECT_ROOT / "reports" / f"{eid}.json"
        if report_file.exists():
            report_file.unlink()


class TestReportPersistence:
    def test_load_nonexistent_report(self):
        from orchestration.analysis_orchestrator import load_report
        assert load_report("EV-NONEXISTENT") is None

    def test_list_reports(self):
        from orchestration.analysis_orchestrator import list_reports
        reports = list_reports()
        assert isinstance(reports, list)
        for eid in reports:
            assert eid.startswith("EV-")
