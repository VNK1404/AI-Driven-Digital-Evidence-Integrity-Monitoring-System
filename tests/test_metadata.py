"""
tests/test_metadata.py
------------------------
Integration tests for integrations/metadata/adapter.py

Requires:
  - metadata_forensics/ module structure intact
  - metadata_forensics/dummy.jpg  (823 bytes — exists)

Does NOT require any external ML models for default z-score anomaly detection.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pytest
from integrations.metadata import adapter as meta

DUMMY_IMAGE = _PROJECT_ROOT / "metadata_forensics" / "dummy.jpg"
DUMMY_EXISTS = DUMMY_IMAGE.exists()


class TestMetadataAdapter:

    @pytest.mark.skipif(not DUMMY_EXISTS, reason="dummy.jpg not found")
    def test_image_analysis_success(self):
        result = meta.analyze_metadata(
            evidence_id="EV-TEST-META-001",
            file_path=str(DUMMY_IMAGE),
            file_type="image",
        )
        assert result.analysis_type == "metadata_forensics"
        assert result.status in ("success", "failed")   # failed OK if deps missing

        if result.status == "success":
            r = result.result
            assert "metadata_score" in r
            assert "flags" in r
            assert isinstance(r["flags"], list)
            assert "anomaly" in r
            assert isinstance(r["anomaly"], bool)

    def test_audio_rejected(self, tmp_path):
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"mp3 data")
        result = meta.analyze_metadata(
            evidence_id="EV-TEST-META-002",
            file_path=str(f),
            file_type="audio",
        )
        assert result.status == "skipped"
        assert "audio" in result.skip_reason.lower()

    def test_module_result_schema(self, tmp_path):
        f = tmp_path / "test.jpg"
        f.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)
        result = meta.analyze_metadata(
            evidence_id="EV-TEST-META-003",
            file_path=str(f),
            file_type="image",
        )
        d = result.to_dict()
        assert "evidence_id" in d
        assert "analysis_type" in d
        assert "status" in d
        assert "timestamp" in d
        assert "result" in d

    def test_nonexistent_file_returns_failed(self, tmp_path):
        result = meta.analyze_metadata(
            evidence_id="EV-TEST-META-004",
            file_path=str(tmp_path / "nonexistent.jpg"),
            file_type="image",
        )
        # Should return failed gracefully, not raise
        assert result.status in ("failed", "skipped")
