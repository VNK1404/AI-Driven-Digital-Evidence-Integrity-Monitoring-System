"""
tests/test_end_to_end.py
--------------------------
End-to-end integration tests for the full forensic pipeline.

Tests the complete analysis_orchestrator.analyze_evidence() flow
using the real test samples that already exist in the repository.

Test samples used
-----------------
- IMAGE : metadata_forensics/dummy.jpg                     (823 bytes)
- VIDEO : deepfake_detector/test_videos/real_sample.mp4    (8 MB)
- VIDEO : deepfake_detector/test_videos/fake_sample.mp4    (8.3 MB)

IMPORTANT
---------
These tests load REAL models (746 MB total) and are therefore slow.
Run selectively:
    pytest tests/test_end_to_end.py -v -k "image"
    pytest tests/test_end_to_end.py -v -k "video"
    pytest tests/test_end_to_end.py -v

They are SKIPPED automatically if the required files or weights are missing.
"""

import sys
import json
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pytest

# ── Sample file paths ──────────────────────────────────────────────────────────
DUMMY_IMAGE    = _PROJECT_ROOT / "metadata_forensics" / "dummy.jpg"
REAL_VIDEO     = _PROJECT_ROOT / "deepfake_detector" / "test_videos" / "real_sample.mp4"
FAKE_VIDEO     = _PROJECT_ROOT / "deepfake_detector" / "test_videos" / "fake_sample.mp4"

# ── Model weight paths ─────────────────────────────────────────────────────────
FORGERY_WEIGHTS = (
    _PROJECT_ROOT / "image_forgery_detection" / "forgery_detection" / "best_resnet50.pth"
)
DEEPFAKE_WEIGHTS = (
    _PROJECT_ROOT / "deepfake_detector" / "models" / "best_model_ffpp.pth"
)

IMAGE_ANALYSIS_POSSIBLE = DUMMY_IMAGE.exists() and FORGERY_WEIGHTS.exists()
VIDEO_ANALYSIS_POSSIBLE = (REAL_VIDEO.exists() or FAKE_VIDEO.exists()) and DEEPFAKE_WEIGHTS.exists()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _validate_report(report_dict: dict, expected_eid: str):
    """Assert common invariants on a ForensicReport dict."""
    assert report_dict["evidence_id"] == expected_eid, "evidence_id mismatch"
    assert report_dict["evidence_id"].startswith("EV-")
    assert report_dict["overall_status"] in ("completed", "partial", "failed")
    assert "file" in report_dict
    assert "analysis_timestamp" in report_dict

    # Every non-None module result must have required keys
    for module_key in ("blockchain", "metadata", "image_forgery", "deepfake", "fake_news"):
        m = report_dict.get(module_key)
        if m is None:
            continue
        assert "evidence_id" in m, f"{module_key} missing evidence_id"
        assert m["evidence_id"] == expected_eid, f"{module_key} evidence_id mismatch"
        assert m["status"] in ("success", "failed", "skipped"), f"{module_key} invalid status"
        assert "timestamp" in m
        assert "result" in m


def _cleanup_report(eid: str):
    """Remove the saved report file after a test."""
    p = _PROJECT_ROOT / "reports" / f"{eid}.json"
    if p.exists():
        try:
            p.unlink()
        except Exception:
            pass


# ── IMAGE end-to-end ───────────────────────────────────────────────────────────

@pytest.mark.skipif(not IMAGE_ANALYSIS_POSSIBLE, reason="dummy.jpg or forgery weights missing")
class TestImageEndToEnd:

    def test_image_analysis_completes(self):
        from orchestration.analysis_orchestrator import analyze_evidence

        t0 = time.perf_counter()
        report = analyze_evidence(str(DUMMY_IMAGE), submitted_by="e2e_test")
        elapsed = time.perf_counter() - t0

        rd = report.to_dict()
        eid = rd["evidence_id"]

        try:
            _validate_report(rd, eid)

            # Image-specific: blockchain and metadata MUST have run
            assert rd["blockchain"] is not None
            assert rd["metadata"] is not None

            # image_forgery MUST have run (weights exist)
            assert rd["image_forgery"] is not None

            # Deepfake MUST NOT have run on an image
            assert rd["deepfake"] is None

            # Evidence ID consistent throughout
            for key in ("blockchain", "metadata", "image_forgery"):
                m = rd.get(key)
                if m:
                    assert m["evidence_id"] == eid

            # Report saved to disk
            report_file = _PROJECT_ROOT / "reports" / f"{eid}.json"
            assert report_file.exists()

            # Report is valid JSON
            with report_file.open() as f:
                saved = json.load(f)
            assert saved["evidence_id"] == eid

            print(f"\n[IMAGE E2E] Completed in {elapsed:.1f}s — {eid}")
            print(f"  blockchain: {rd['blockchain']['status']}")
            print(f"  metadata:   {rd['metadata']['status']}")
            print(f"  forgery:    {rd['image_forgery']['status']}")

        finally:
            _cleanup_report(eid)

    def test_image_blockchain_sha256_matches_evidence(self):
        from orchestration.analysis_orchestrator import analyze_evidence

        report = analyze_evidence(str(DUMMY_IMAGE))
        rd = report.to_dict()
        eid = rd["evidence_id"]

        try:
            file_sha256 = rd["file"]["sha256"]
            if rd["blockchain"] and rd["blockchain"]["status"] == "success":
                bc_sha256 = rd["blockchain"]["result"]["sha256"]
                assert file_sha256 == bc_sha256, "SHA-256 mismatch between file and blockchain"
        finally:
            _cleanup_report(eid)


# ── VIDEO end-to-end ───────────────────────────────────────────────────────────

@pytest.mark.skipif(not VIDEO_ANALYSIS_POSSIBLE, reason="Test videos or deepfake weights missing")
class TestVideoEndToEnd:

    @pytest.mark.parametrize("video_path", [
        pytest.param(REAL_VIDEO, id="real_video",
                     marks=pytest.mark.skipif(not REAL_VIDEO.exists(), reason="real_sample.mp4 missing")),
        pytest.param(FAKE_VIDEO, id="fake_video",
                     marks=pytest.mark.skipif(not FAKE_VIDEO.exists(), reason="fake_sample.mp4 missing")),
    ])
    def test_video_analysis_completes(self, video_path):
        from orchestration.analysis_orchestrator import analyze_evidence

        t0 = time.perf_counter()
        report = analyze_evidence(str(video_path), submitted_by="e2e_test")
        elapsed = time.perf_counter() - t0

        rd = report.to_dict()
        eid = rd["evidence_id"]

        try:
            _validate_report(rd, eid)

            # Video-specific: deepfake MUST have run
            assert rd["deepfake"] is not None

            # image_forgery MUST NOT run on video
            assert rd["image_forgery"] is None

            # fake_news MUST NOT run on video
            assert rd.get("fake_news") is None or rd["fake_news"]["status"] == "skipped"

            if rd["deepfake"]["status"] == "success":
                r = rd["deepfake"]["result"]
                assert r["prediction"] in ("REAL", "FAKE")
                assert r["frames_analyzed"] == 15

            print(f"\n[VIDEO E2E] {video_path.name} in {elapsed:.1f}s — {eid}")
            print(f"  deepfake: {rd['deepfake']['status']}")
            if rd["deepfake"]["status"] == "success":
                print(f"  verdict:  {rd['deepfake']['result']['prediction']}")
        finally:
            _cleanup_report(eid)


# ── Fault isolation ────────────────────────────────────────────────────────────

class TestFaultIsolation:
    """
    Verify that one module failure doesn't crash the whole pipeline.
    Uses mocking so no model loading is needed.
    """

    def test_metadata_crash_isolated(self, tmp_path):
        from unittest.mock import patch
        from orchestration.analysis_orchestrator import analyze_evidence

        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 50)

        with patch("integrations.metadata.adapter.analyze_metadata",
                   side_effect=RuntimeError("metadata explosion")):
            with patch("integrations.blockchain.adapter.register_evidence") as mock_bc:
                from schemas.results import ModuleResult

                def bc_ok(*a, **kw):
                    eid = kw.get("evidence_id", a[0] if a else "EV-TEST")
                    return ModuleResult(
                        evidence_id=eid,
                        analysis_type="blockchain",
                        status="success",
                        result={"sha256": "abc", "integrity_status": "REGISTERED",
                                "chain_length": 1, "chain_valid": True,
                                "registered_at": "2026-01-01T00:00:00Z"},
                    )

                mock_bc.side_effect = bc_ok

                with patch("integrations.image_forgery.adapter.analyze_image_forgery") as mock_fg:
                    def fg_ok(*a, **kw):
                        eid = kw.get("evidence_id", a[0] if a else "EV-TEST")
                        return ModuleResult(
                            evidence_id=eid,
                            analysis_type="image_forgery",
                            status="success",
                            result={"prediction": "AUTHENTIC", "confidence": 0.9,
                                    "scores": {}, "model": "test"},
                        )
                    mock_fg.side_effect = fg_ok

                    with patch("integrations.fake_news.adapter.is_news_content_applicable",
                               return_value=False):
                        report = analyze_evidence(str(img))
                        rd = report.to_dict()
                        eid = rd["evidence_id"]

                        try:
                            assert rd["blockchain"]["status"] == "success"
                            assert rd["metadata"]["status"] == "failed"
                            assert rd["image_forgery"]["status"] == "success"
                            assert rd["overall_status"] == "partial"
                        finally:
                            _cleanup_report(eid)
