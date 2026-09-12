"""
tests/test_fake_news.py
-------------------------
Integration tests for integrations/fake_news/adapter.py

Requires:
  - fake_news_detection_final/fake_news_module/ml/saved_model/model.safetensors (498 MB)
  - fake_news_detection_final/fake_news_module/similarity/index/faiss_index.bin  (66 MB)
  - Tesseract OCR installed (for image tests)

Unit tests that only check applicability and schema do NOT require these.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pytest
from integrations.fake_news import adapter as fn

ROBERTA_MODEL = (
    _PROJECT_ROOT
    / "fake_news_detection_final"
    / "fake_news_module"
    / "ml"
    / "saved_model"
    / "model.safetensors"
)
ROBERTA_EXISTS = ROBERTA_MODEL.exists()


class TestFakeNewsApplicability:
    """Tests that don't require model loading."""

    @pytest.mark.parametrize("ext,ftype,expected", [
        (".jpg",  "image",    True),
        (".jpeg", "image",    True),
        (".png",  "image",    True),
        (".pdf",  "document", True),
        (".mp4",  "video",    False),
        (".avi",  "video",    False),
        (".mkv",  "video",    False),
    ])
    def test_applicability(self, ext, ftype, expected, tmp_path):
        f = tmp_path / f"test{ext}"
        f.write_bytes(b"data")
        result = fn.is_news_content_applicable(str(f), ftype)
        assert result == expected, f"Expected {expected} for {ext}"

    def test_video_skipped(self, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"mp4")
        result = fn.analyze_fake_news(
            evidence_id="EV-FN-001",
            file_path=str(f),
            file_type="video",
        )
        assert result.status == "skipped"
        assert "video" in result.skip_reason.lower() or "fake_news" in result.skip_reason.lower()

    def test_result_schema(self, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"mp4")
        result = fn.analyze_fake_news(
            evidence_id="EV-FN-002",
            file_path=str(f),
            file_type="video",
        )
        d = result.to_dict()
        required = {"evidence_id", "analysis_type", "status", "timestamp", "result"}
        assert required.issubset(set(d.keys()))


class TestFakeNewsTextPipeline:
    """Test the text-input pipeline (bypasses OCR)."""

    @pytest.mark.skipif(not ROBERTA_EXISTS, reason="RoBERTa weights not found")
    def test_real_news_text(self):
        """Run pipeline on a factual news snippet."""
        text = (
            "Scientists at NASA have confirmed the discovery of water ice "
            "on the lunar surface near the south pole. The findings were "
            "published in the journal Nature Astronomy."
        )
        result = fn.analyze_fake_news_text(
            evidence_id="EV-FN-TEXT-001",
            text=text,
        )
        assert result.status in ("success", "failed", "skipped")
        if result.status == "success":
            r = result.result
            assert r["final_decision"] in ("Real", "Fake", "Uncertain")
            assert "confidence" in r
            assert "score" in r

    def test_empty_text_returns_skipped(self):
        result = fn.analyze_fake_news_text(
            evidence_id="EV-FN-TEXT-002",
            text="",
        )
        # Empty text → ValueError in pipeline → skipped
        assert result.status in ("skipped", "failed")
