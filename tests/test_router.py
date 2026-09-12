"""
tests/test_router.py
---------------------
Unit tests for orchestration/evidence_router.py

These tests do NOT require model loading or network access.
They verify that file extensions are correctly mapped to file types
and that routing decisions are correct.
"""

import sys
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pytest
from orchestration.evidence_router import route_evidence, describe_routing


# ── Extension → file_type mapping ─────────────────────────────────────────────

class TestFileTypeDetection:
    """Verify that extensions are correctly categorised."""

    @pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"])
    def test_image_extensions(self, ext, tmp_path):
        f = tmp_path / f"test{ext}"
        f.write_bytes(b"fake image data")
        d = route_evidence(str(f))
        assert d.file_type == "image", f"Expected image for {ext}, got {d.file_type}"
        assert d.is_supported

    @pytest.mark.parametrize("ext", [".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv", ".flv"])
    def test_video_extensions(self, ext, tmp_path):
        f = tmp_path / f"test{ext}"
        f.write_bytes(b"fake video data")
        d = route_evidence(str(f))
        assert d.file_type == "video"
        assert d.is_supported

    @pytest.mark.parametrize("ext", [".pdf"])
    def test_document_extensions(self, ext, tmp_path):
        f = tmp_path / f"test{ext}"
        f.write_bytes(b"fake pdf data")
        d = route_evidence(str(f))
        assert d.file_type == "document"
        assert d.is_supported

    @pytest.mark.parametrize("ext", [".mp3", ".wav", ".aac", ".flac"])
    def test_audio_rejected(self, ext, tmp_path):
        f = tmp_path / f"test{ext}"
        f.write_bytes(b"fake audio")
        d = route_evidence(str(f))
        assert not d.is_supported
        assert d.file_type == "audio"

    @pytest.mark.parametrize("ext", [".exe", ".zip", ".docx", ".xlsx"])
    def test_unsupported_extensions(self, ext, tmp_path):
        f = tmp_path / f"test{ext}"
        f.write_bytes(b"data")
        d = route_evidence(str(f))
        assert not d.is_supported
        assert d.reject_reason


# ── Routing decisions ─────────────────────────────────────────────────────────

class TestRoutingDecisions:
    """Verify that routing decisions set the correct module flags."""

    def test_image_routing(self, tmp_path):
        f = (tmp_path / "photo.jpg")
        f.write_bytes(b"jpg")
        d = route_evidence(str(f))
        assert d.run_blockchain is True
        assert d.run_metadata is True
        assert d.run_image_forgery is True
        assert d.run_deepfake is False
        assert d.run_fake_news is True     # conditional

    def test_video_routing(self, tmp_path):
        f = (tmp_path / "clip.mp4")
        f.write_bytes(b"mp4")
        d = route_evidence(str(f))
        assert d.run_blockchain is True
        assert d.run_metadata is True
        assert d.run_image_forgery is False
        assert d.run_deepfake is True
        assert d.run_fake_news is False

    def test_document_routing(self, tmp_path):
        f = (tmp_path / "doc.pdf")
        f.write_bytes(b"pdf")
        d = route_evidence(str(f))
        assert d.run_blockchain is True
        assert d.run_metadata is True
        assert d.run_image_forgery is False
        assert d.run_deepfake is False
        assert d.run_fake_news is True     # conditional

    def test_unsupported_routing(self, tmp_path):
        f = (tmp_path / "archive.zip")
        f.write_bytes(b"zip")
        d = route_evidence(str(f))
        assert d.is_supported is False
        assert d.run_blockchain is False
        assert d.run_deepfake is False

    def test_audio_routing(self, tmp_path):
        f = (tmp_path / "audio.mp3")
        f.write_bytes(b"mp3")
        d = route_evidence(str(f))
        assert d.is_supported is False
        assert "audio" in d.reject_reason.lower()

    def test_describe_routing_image(self, tmp_path):
        f = (tmp_path / "photo.png")
        f.write_bytes(b"png")
        d = route_evidence(str(f))
        desc = describe_routing(d)
        assert "IMAGE" in desc
        assert "blockchain" in desc
        assert "image_forgery" in desc

    def test_describe_routing_rejected(self, tmp_path):
        f = (tmp_path / "song.wav")
        f.write_bytes(b"wav")
        d = route_evidence(str(f))
        desc = describe_routing(d)
        assert "REJECTED" in desc
