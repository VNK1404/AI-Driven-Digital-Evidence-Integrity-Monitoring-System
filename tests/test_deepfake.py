"""
tests/test_deepfake.py
------------------------
Integration tests for integrations/deepfake/adapter.py

Requires:
  - deepfake_detector/models/best_model_ffpp.pth   (66 MB)
  - deepfake_detector/test_videos/real_sample.mp4  (8 MB)
  - deepfake_detector/test_videos/fake_sample.mp4  (8.3 MB)

NOTE: Model first-load takes ~15s on CPU.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pytest
from integrations.deepfake import adapter as df

MODEL_EXISTS = (
    _PROJECT_ROOT / "deepfake_detector" / "models" / "best_model_ffpp.pth"
).exists()

REAL_VIDEO = _PROJECT_ROOT / "deepfake_detector" / "test_videos" / "real_sample.mp4"
FAKE_VIDEO = _PROJECT_ROOT / "deepfake_detector" / "test_videos" / "fake_sample.mp4"
VIDEOS_EXIST = REAL_VIDEO.exists() and FAKE_VIDEO.exists()


class TestDeepfakeAdapter:

    def test_non_video_skipped(self, tmp_path):
        f = tmp_path / "photo.jpg"
        f.write_bytes(b"jpg")
        result = df.analyze_deepfake(
            evidence_id="EV-DF-001",
            file_path=str(f),
            file_type="image",
        )
        assert result.status == "skipped"
        assert "video" in result.skip_reason.lower()

    def test_document_skipped(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"pdf")
        result = df.analyze_deepfake(
            evidence_id="EV-DF-002",
            file_path=str(f),
            file_type="document",
        )
        assert result.status == "skipped"

    @pytest.mark.skipif(not MODEL_EXISTS, reason="Model weights not found")
    @pytest.mark.skipif(not VIDEOS_EXIST, reason="Test videos not found")
    def test_real_video_detection(self):
        result = df.analyze_deepfake(
            evidence_id="EV-DF-003",
            file_path=str(REAL_VIDEO),
            file_type="video",
        )
        assert result.status in ("success", "failed")
        if result.status == "success":
            r = result.result
            assert r["prediction"] in ("REAL", "FAKE")
            assert 0.0 <= r["confidence_pct"] <= 100.0
            assert r["frames_analyzed"] == 15
            assert "EfficientNet-B3" in r["model"]

    @pytest.mark.skipif(not MODEL_EXISTS, reason="Model weights not found")
    @pytest.mark.skipif(not VIDEOS_EXIST, reason="Test videos not found")
    def test_fake_video_detection(self):
        result = df.analyze_deepfake(
            evidence_id="EV-DF-004",
            file_path=str(FAKE_VIDEO),
            file_type="video",
        )
        assert result.status in ("success", "failed")
        if result.status == "success":
            r = result.result
            assert r["prediction"] in ("REAL", "FAKE")

    def test_nonexistent_video_returns_failed(self, tmp_path):
        result = df.analyze_deepfake(
            evidence_id="EV-DF-005",
            file_path=str(tmp_path / "ghost.mp4"),
            file_type="video",
        )
        assert result.status in ("failed", "skipped")
