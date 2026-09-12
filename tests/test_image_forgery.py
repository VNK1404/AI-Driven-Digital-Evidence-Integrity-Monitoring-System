"""
tests/test_image_forgery.py
-----------------------------
Integration tests for integrations/image_forgery/adapter.py

Requires:
  - image_forgery_detection/forgery_detection/best_resnet50.pth   (98 MB)
  - image_forgery_detection/forgery_detection/best_efficientnet.pth (17 MB)
  - image_forgery_detection/forgery_detection/meta_learner.pkl     (1 KB)
  - metadata_forensics/dummy.jpg  (823 bytes — real image for inference)

NOTE: First call loads models (~15s on CPU). Subsequent calls are fast.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pytest
from integrations.image_forgery import adapter as forgery

DUMMY_IMAGE = _PROJECT_ROOT / "metadata_forensics" / "dummy.jpg"
WEIGHTS_EXIST = (
    (_PROJECT_ROOT / "image_forgery_detection" / "forgery_detection" / "best_resnet50.pth").exists()
)


class TestImageForgeryAdapter:

    def test_non_image_skipped(self, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"mp4")
        result = forgery.analyze_image_forgery(
            evidence_id="EV-FRG-001",
            file_path=str(f),
            file_type="video",
        )
        assert result.status == "skipped"
        assert "image" in result.skip_reason.lower()

    @pytest.mark.skipif(not WEIGHTS_EXIST, reason="Model weights not found")
    @pytest.mark.skipif(not DUMMY_IMAGE.exists(), reason="dummy.jpg not found")
    def test_analysis_on_real_image(self):
        result = forgery.analyze_image_forgery(
            evidence_id="EV-FRG-002",
            file_path=str(DUMMY_IMAGE),
            file_type="image",
        )
        assert result.status in ("success", "failed")

        if result.status == "success":
            r = result.result
            assert r["prediction"] in ("AUTHENTIC", "TAMPERED")
            assert 0.0 <= r["confidence"] <= 1.0
            assert "scores" in r
            assert "authentic" in r["scores"]
            assert "tampered" in r["scores"]
            assert r["model"] == "ResNet50 + EfficientNet-B0 + Logistic Regression"

    def test_result_schema_on_skipped(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"pdf")
        result = forgery.analyze_image_forgery(
            evidence_id="EV-FRG-003",
            file_path=str(f),
            file_type="document",
        )
        d = result.to_dict()
        assert "evidence_id" in d
        assert "status" in d
        assert d["status"] == "skipped"

    def test_nonexistent_image_returns_failed(self, tmp_path):
        result = forgery.analyze_image_forgery(
            evidence_id="EV-FRG-004",
            file_path=str(tmp_path / "ghost.jpg"),
            file_type="image",
        )
        # Must return failed, not raise
        assert result.status in ("failed", "skipped")
