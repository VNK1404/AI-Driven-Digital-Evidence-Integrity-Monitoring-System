"""
tests/test_blockchain.py
--------------------------
Integration tests for integrations/blockchain/adapter.py

These tests USE the actual blockchain module's SQLite database.
They do NOT require any ML models or network access.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import pytest
from integrations.blockchain import adapter as bc


class TestBlockchainAdapter:
    """Tests that use a real (but test-isolated) evidence file."""

    @pytest.fixture
    def dummy_image(self, tmp_path):
        """Create a minimal JPEG-like dummy image for testing."""
        f = tmp_path / "dummy_evidence.jpg"
        f.write_bytes(b"\xff\xd8\xff" + b"\x00" * 200)
        return f

    def test_register_evidence_success(self, dummy_image):
        from schemas.evidence import compute_sha256, _generate_evidence_id
        eid = _generate_evidence_id()
        sha256 = compute_sha256(dummy_image)

        result = bc.register_evidence(
            evidence_id=eid,
            file_path=str(dummy_image),
            sha256=sha256,
            file_type="image",
            file_name=dummy_image.name,
            file_size=dummy_image.stat().st_size,
            user="test_officer",
        )
        assert result.status == "success"
        assert result.evidence_id == eid
        assert result.result["sha256"] == sha256
        assert result.result["integrity_status"] == "REGISTERED"

    def test_verify_integrity_verified(self, dummy_image):
        from schemas.evidence import compute_sha256, _generate_evidence_id
        eid = _generate_evidence_id()
        sha256 = compute_sha256(dummy_image)

        # Register first
        bc.register_evidence(
            evidence_id=eid,
            file_path=str(dummy_image),
            sha256=sha256,
            file_type="image",
            file_name=dummy_image.name,
            file_size=dummy_image.stat().st_size,
            user="system",
        )

        # Verify — should pass because file is unchanged
        result = bc.verify_integrity(evidence_id=eid, file_path=str(dummy_image))
        assert result.status == "success"
        assert result.result["integrity_status"] == "VERIFIED"
        assert result.result["tampered"] is False

    def test_verify_integrity_tampered(self, dummy_image):
        from schemas.evidence import compute_sha256, _generate_evidence_id
        eid = _generate_evidence_id()
        sha256 = compute_sha256(dummy_image)

        bc.register_evidence(
            evidence_id=eid,
            file_path=str(dummy_image),
            sha256=sha256,
            file_type="image",
            file_name=dummy_image.name,
            file_size=dummy_image.stat().st_size,
            user="system",
        )

        # Tamper the file
        with dummy_image.open("ab") as f:
            f.write(b"TAMPERED DATA")

        result = bc.verify_integrity(evidence_id=eid, file_path=str(dummy_image))
        assert result.status == "success"
        assert result.result["integrity_status"] == "TAMPERED"
        assert result.result["tampered"] is True

    def test_custody_log_empty_on_unknown_id(self):
        log = bc.get_custody_log("EV-NONEXISTENT")
        assert isinstance(log, list)
        # Should return empty list, not crash

    def test_get_chain_status(self):
        status = bc.get_chain_status()
        assert "chain_length" in status
        assert "chain_valid" in status
        assert status["chain_valid"] is True   # genesis block always valid
        assert status["chain_length"] >= 1     # at least genesis

    def test_log_custody_event_silent_fail(self):
        """log_custody_event must not raise even with invalid evidence_id."""
        # Should log warning but not crash
        bc.log_custody_event(
            evidence_id="EV-DOESNOTEXIST",
            action="access",
            user="tester",
        )
