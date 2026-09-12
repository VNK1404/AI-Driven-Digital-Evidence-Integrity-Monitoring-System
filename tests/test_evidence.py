"""
tests/test_evidence.py
------------------------
Unit tests for schemas/evidence.py

Tests EvidenceObject creation, SHA-256 computation, evidence_id format,
and serialisation.  No model loading or network access required.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pytest
from schemas.evidence import EvidenceObject, compute_sha256, _generate_evidence_id


class TestEvidenceIdGeneration:
    def test_format(self):
        eid = _generate_evidence_id()
        assert eid.startswith("EV-")
        assert len(eid) == 9       # "EV-" + 6 digits

    def test_uniqueness(self):
        ids = {_generate_evidence_id() for _ in range(100)}
        # With 1 million possible IDs, 100 draws should all be unique
        assert len(ids) == 100


class TestComputeSha256:
    def test_known_hash(self, tmp_path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        digest = compute_sha256(f)
        # SHA-256 of "hello world"
        assert digest == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        assert len(digest) == 64

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        digest = compute_sha256(f)
        # SHA-256 of empty string
        assert digest == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            compute_sha256(tmp_path / "nonexistent.jpg")


class TestEvidenceObject:
    def test_image_creation(self, tmp_path):
        f = tmp_path / "test.jpg"
        f.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)  # JPEG magic bytes
        ev = EvidenceObject(file_path=f, file_type="image")

        assert ev.evidence_id.startswith("EV-")
        assert ev.file_name == "test.jpg"
        assert ev.file_extension == ".jpg"
        assert ev.file_type == "image"
        assert ev.size_bytes == 103
        assert len(ev.sha256) == 64
        assert ev.mime_type == "image/jpeg"
        assert ev.submitted_by == "system"

    def test_video_creation(self, tmp_path):
        f = tmp_path / "clip.mp4"
        f.write_bytes(b"\x00" * 50)
        ev = EvidenceObject(file_path=f, file_type="video", submitted_by="officer")

        assert ev.file_type == "video"
        assert ev.mime_type == "video/mp4"
        assert ev.submitted_by == "officer"

    def test_document_creation(self, tmp_path):
        f = tmp_path / "report.pdf"
        f.write_bytes(b"%PDF-1.4 test")
        ev = EvidenceObject(file_path=f, file_type="document")

        assert ev.file_type == "document"
        assert ev.mime_type == "application/pdf"

    def test_sha256_deterministic(self, tmp_path):
        f = tmp_path / "data.png"
        f.write_bytes(b"consistent data")
        ev1 = EvidenceObject(file_path=f, file_type="image")
        ev2 = EvidenceObject(file_path=f, file_type="image")
        assert ev1.sha256 == ev2.sha256

    def test_to_dict_keys(self, tmp_path):
        f = tmp_path / "test.jpg"
        f.write_bytes(b"data")
        ev = EvidenceObject(file_path=f, file_type="image")
        d = ev.to_dict()

        expected_keys = {
            "evidence_id", "file_name", "file_path", "file_type",
            "file_extension", "size_bytes", "sha256", "mime_type",
            "submitted_by", "submitted_at",
        }
        assert expected_keys == set(d.keys())

    def test_file_not_found(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            EvidenceObject(file_path=tmp_path / "nonexistent.jpg", file_type="image")
