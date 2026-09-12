"""
test_module.py
--------------
Complete test suite for the Cryptographic Integrity Verification
& Blockchain Evidence Ledger Module.

Tests every component:
  1. Hashing       - SHA-256 for all file types
  2. Blockchain    - Block creation, chain validation
  3. Verify        - VERIFIED / TAMPERED detection
  4. Database      - Save, retrieve, update records
  5. Custody       - Audit logs, tamper alerts
  6. API Endpoints - All 7 Flask endpoints

USAGE:
    # Make sure Flask server is running first in another terminal!
    python test_module.py
"""

import os
import sys
import json
import tempfile
import requests
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Colours for terminal output ───────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

API_BASE = "http://127.0.0.1:5001"

passed = 0
failed = 0


def ok(msg):
    global passed
    passed += 1
    print(f"  {GREEN}✅ PASS{RESET}  {msg}")


def fail(msg, reason=""):
    global failed
    failed += 1
    print(f"  {RED}❌ FAIL{RESET}  {msg}")
    if reason:
        print(f"         {RED}→ {reason}{RESET}")


def section(title):
    print(f"\n{BOLD}{BLUE}{'─'*55}{RESET}")
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(f"{BOLD}{BLUE}{'─'*55}{RESET}")


def create_temp_file(content: bytes, suffix: str) -> str:
    """Create a temporary file with given content and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.close()
    return tmp.name


# ══════════════════════════════════════════════════════
# TEST 1: HASHING
# ══════════════════════════════════════════════════════
def test_hashing():
    section("TEST 1: Hashing Module (hashing.py)")
    from integrity.hashing import generate_hash, get_file_type, is_supported, get_file_metadata

    # Test 1.1 — Hash a real file
    try:
        f = create_temp_file(b"This is test evidence content.", ".txt")
        h = generate_hash(f)
        assert len(h) == 64, "Hash should be 64 hex chars"
        ok("SHA-256 hash generated (64 hex chars)")
        os.unlink(f)
    except Exception as e:
        fail("SHA-256 hash generation", str(e))

    # Test 1.2 — Same content = same hash
    try:
        f1 = create_temp_file(b"identical content", ".txt")
        f2 = create_temp_file(b"identical content", ".txt")
        assert generate_hash(f1) == generate_hash(f2)
        ok("Same content produces identical hashes")
        os.unlink(f1); os.unlink(f2)
    except Exception as e:
        fail("Identical content hash match", str(e))

    # Test 1.3 — Different content = different hash
    try:
        f1 = create_temp_file(b"original content", ".txt")
        f2 = create_temp_file(b"tampered content", ".txt")
        assert generate_hash(f1) != generate_hash(f2)
        ok("Different content produces different hashes")
        os.unlink(f1); os.unlink(f2)
    except Exception as e:
        fail("Different content hash mismatch", str(e))

    # Test 1.4 — File not found error
    try:
        generate_hash("/nonexistent/file.jpg")
        fail("FileNotFoundError not raised for missing file")
    except FileNotFoundError:
        ok("FileNotFoundError raised for missing file")
    except Exception as e:
        fail("FileNotFoundError for missing file", str(e))

    # Test 1.5 — File type detection
    try:
        assert get_file_type("photo.jpg")   == "image"
        assert get_file_type("video.mp4")   == "video"
        assert get_file_type("audio.mp3")   == "audio"
        assert get_file_type("report.pdf")  == "document"
        assert get_file_type("unknown.xyz") == "unknown"
        ok("File type detection (image/video/audio/document/unknown)")
    except Exception as e:
        fail("File type detection", str(e))

    # Test 1.6 — Supported extension check
    try:
        assert is_supported("test.jpg")  == True
        assert is_supported("test.mp4")  == True
        assert is_supported("test.mp3")  == True
        assert is_supported("test.pdf")  == True
        assert is_supported("test.xyz")  == False
        ok("Supported extension validation")
    except Exception as e:
        fail("Supported extension check", str(e))

    # Test 1.7 — Metadata extraction
    try:
        f = create_temp_file(b"metadata test content", ".jpg")
        meta = get_file_metadata(f)
        assert "file_name"  in meta
        assert "file_size"  in meta
        assert "file_type"  in meta
        assert meta["file_type"] == "image"
        assert meta["file_size"] > 0
        ok("File metadata extraction (name, size, type)")
        os.unlink(f)
    except Exception as e:
        fail("File metadata extraction", str(e))


# ══════════════════════════════════════════════════════
# TEST 2: BLOCKCHAIN
# ══════════════════════════════════════════════════════
def test_blockchain():
    section("TEST 2: Blockchain Module (blockchain.py)")
    from integrity.blockchain import Blockchain

    # Fresh blockchain instance for testing
    bc = Blockchain()

    # Test 2.1 — Genesis block exists
    try:
        assert len(bc.chain) == 1
        assert bc.chain[0].index == 0
        assert bc.chain[0].previous_hash == "0"
        ok("Genesis block created at index 0")
    except Exception as e:
        fail("Genesis block creation", str(e))

    # Test 2.2 — Add a block
    try:
        block = bc.add_approved_block(
            {"evidence_id": "TEST001", "hash_value": "abc123", "file_type": "image"},
            approved_by="test_officer"
        )
        assert block.index == 1
        assert len(bc.chain) == 2
        ok("Block added at index 1")
    except Exception as e:
        fail("Adding block to chain", str(e))

    # Test 2.3 — Block links correctly
    try:
        assert bc.chain[1].previous_hash == bc.chain[0].hash
        ok("Block linked to previous block hash")
    except Exception as e:
        fail("Block linkage verification", str(e))

    # Test 2.4 — Chain is valid
    try:
        assert bc.is_chain_valid() == True
        ok("Chain validation passes on clean chain")
    except Exception as e:
        fail("Chain validation", str(e))

    # Test 2.5 — Tampering detected
    try:
        bc.chain[1].evidence_data["hash_value"] = "tampered_hash"
        assert bc.is_chain_valid() == False
        ok("Tampering detected — chain validation fails after modification")
        # Restore for further tests
        bc.chain[1].evidence_data["hash_value"] = "abc123"
    except Exception as e:
        fail("Tamper detection in chain", str(e))

    # Test 2.6 — Get latest block
    try:
        latest = bc.get_latest_block()
        assert latest.index == 1
        ok("get_latest_block() returns correct block")
    except Exception as e:
        fail("get_latest_block()", str(e))

    # Test 2.7 — Find block by evidence_id
    try:
        found = bc.find_block("TEST001")
        assert found is not None
        assert found["evidence_data"]["evidence_id"] == "TEST001"
        ok("find_block() locates block by evidence_id")
    except Exception as e:
        fail("find_block() by evidence_id", str(e))

    # Test 2.8 — Find nonexistent block
    try:
        not_found = bc.find_block("NONEXISTENT")
        assert not_found is None
        ok("find_block() returns None for unknown evidence_id")
    except Exception as e:
        fail("find_block() for unknown ID", str(e))

    # Test 2.9 — Multiple blocks
    try:
        for i in range(2, 6):
            bc.add_approved_block(
                {"evidence_id": f"TEST00{i}", "hash_value": f"hash_{i}"},
                approved_by="officer"
            )
        assert len(bc.chain) == 6
        assert bc.is_chain_valid() == True
        ok("Multiple blocks added, chain remains valid")
    except Exception as e:
        fail("Multiple block addition", str(e))


# ══════════════════════════════════════════════════════
# TEST 3: INTEGRITY VERIFICATION
# ══════════════════════════════════════════════════════
def test_verify():
    section("TEST 3: Integrity Verification (verify.py)")
    from integrity.hashing import generate_hash
    from integrity.verify  import verify_integrity, STATUS_VERIFIED, STATUS_TAMPERED, STATUS_ERROR

    # Test 3.1 — VERIFIED: same file
    try:
        f = create_temp_file(b"original untampered content", ".pdf")
        h = generate_hash(f)
        result = verify_integrity(f, h)
        assert result["status"]   == STATUS_VERIFIED
        assert result["tampered"] == False
        ok("VERIFIED returned for matching hash")
        os.unlink(f)
    except Exception as e:
        fail("VERIFIED status check", str(e))

    # Test 3.2 — TAMPERED: modified content
    try:
        f = create_temp_file(b"original content", ".jpg")
        h = generate_hash(f)
        # Modify the file
        with open(f, "wb") as fp:
            fp.write(b"tampered content here!")
        result = verify_integrity(f, h)
        assert result["status"]   == STATUS_TAMPERED
        assert result["tampered"] == True
        ok("TAMPERED returned for modified file")
        os.unlink(f)
    except Exception as e:
        fail("TAMPERED status check", str(e))

    # Test 3.3 — ERROR: file not found
    try:
        result = verify_integrity("/does/not/exist.mp4", "somehash")
        assert result["status"] == STATUS_ERROR
        ok("ERROR returned for missing file")
    except Exception as e:
        fail("ERROR status for missing file", str(e))

    # Test 3.4 — current_hash and stored_hash returned
    try:
        f = create_temp_file(b"check hash fields", ".txt")
        h = generate_hash(f)
        result = verify_integrity(f, h)
        assert result["current_hash"] == h
        assert result["stored_hash"]  == h
        ok("current_hash and stored_hash returned in response")
        os.unlink(f)
    except Exception as e:
        fail("Hash fields in verify response", str(e))

    # Test 3.5 — Message included
    try:
        f = create_temp_file(b"message test", ".txt")
        h = generate_hash(f)
        result = verify_integrity(f, h)
        assert "message" in result
        assert len(result["message"]) > 0
        ok("Descriptive message included in verify response")
        os.unlink(f)
    except Exception as e:
        fail("Message in verify response", str(e))


# ══════════════════════════════════════════════════════
# TEST 4: DATABASE
# ══════════════════════════════════════════════════════
def test_database():
    section("TEST 4: Database Module (db.py)")
    from database.db import (
        init_db, save_evidence, get_evidence,
        get_all_evidence, update_evidence_status
    )

    init_db()

    ts = datetime.utcnow().isoformat()
    test_id = f"DBTEST_{int(datetime.utcnow().timestamp())}"

    # Test 4.1 — Save evidence
    try:
        save_evidence(test_id, "test.jpg", "/tmp/test.jpg",
                      "image", 1024, "hashvalue123", "tester", ts)
        ok("Evidence record saved to database")
    except Exception as e:
        fail("Save evidence to DB", str(e))

    # Test 4.2 — Retrieve evidence
    try:
        rec = get_evidence(test_id)
        assert rec is not None
        assert rec["evidence_id"] == test_id
        assert rec["file_type"]   == "image"
        assert rec["status"]      == "pending"
        ok("Evidence record retrieved by ID")
    except Exception as e:
        fail("Retrieve evidence from DB", str(e))

    # Test 4.3 — Update status
    try:
        update_evidence_status(test_id, "verified")
        rec = get_evidence(test_id)
        assert rec["status"] == "verified"
        ok("Evidence status updated to 'verified'")
    except Exception as e:
        fail("Update evidence status", str(e))

    # Test 4.4 — Get nonexistent record
    try:
        rec = get_evidence("NONEXISTENT_ID_XYZ")
        assert rec is None
        ok("None returned for nonexistent evidence ID")
    except Exception as e:
        fail("None for nonexistent record", str(e))

    # Test 4.5 — Get all evidence
    try:
        all_recs = get_all_evidence()
        assert isinstance(all_recs, list)
        assert len(all_recs) >= 1
        ok(f"get_all_evidence() returns list ({len(all_recs)} records)")
    except Exception as e:
        fail("get_all_evidence()", str(e))

    # Test 4.6 — Filter by type
    try:
        images = get_all_evidence(file_type="image")
        assert all(r["file_type"] == "image" for r in images)
        ok("Filter evidence by file_type works")
    except Exception as e:
        fail("Filter by file_type", str(e))

    # Test 4.7 — Filter by status
    try:
        verified = get_all_evidence(status="verified")
        assert all(r["status"] == "verified" for r in verified)
        ok("Filter evidence by status works")
    except Exception as e:
        fail("Filter by status", str(e))


# ══════════════════════════════════════════════════════
# TEST 5: CUSTODY LOGS
# ══════════════════════════════════════════════════════
def test_custody():
    section("TEST 5: Custody & Alerts (custody.py)")
    from database.db import init_db, save_evidence
    from integrity.custody import (
        log_custody_event, get_custody_log,
        save_tamper_alert, get_all_alerts,
        ACTION_UPLOAD, ACTION_VERIFY, ACTION_APPROVE
    )

    init_db()
    ts      = datetime.utcnow().isoformat()
    test_id = f"CUST_{int(datetime.utcnow().timestamp())}"
    save_evidence(test_id, "custody_test.mp4", "/tmp/x.mp4",
                  "video", 2048, "hash_cust", "tester", ts)

    # Test 5.1 — Log upload event
    try:
        event = log_custody_event(test_id, ACTION_UPLOAD, "officer_test", "Uploaded via test")
        assert event["evidence_id"] == test_id
        assert event["action"]      == ACTION_UPLOAD
        ok("Upload custody event logged")
    except Exception as e:
        fail("Log upload custody event", str(e))

    # Test 5.2 — Log verify event
    try:
        log_custody_event(test_id, ACTION_VERIFY, "officer_test", "Verified OK")
        ok("Verify custody event logged")
    except Exception as e:
        fail("Log verify custody event", str(e))

    # Test 5.3 — Log approve event
    try:
        log_custody_event(test_id, ACTION_APPROVE, "officer_test", "Approved for blockchain")
        ok("Approve custody event logged")
    except Exception as e:
        fail("Log approve custody event", str(e))

    # Test 5.4 — Retrieve custody log
    try:
        log = get_custody_log(test_id)
        assert len(log) >= 3
        actions = [e["action"] for e in log]
        assert ACTION_UPLOAD  in actions
        assert ACTION_VERIFY  in actions
        assert ACTION_APPROVE in actions
        ok(f"Custody log retrieved ({len(log)} events)")
    except Exception as e:
        fail("Retrieve custody log", str(e))

    # Test 5.5 — Save tamper alert
    try:
        alert = save_tamper_alert(
            test_id, "custody_test.mp4", "video",
            "original_hash_123", "tampered_hash_456", "officer_test"
        )
        assert alert["evidence_id"]  == test_id
        assert alert["stored_hash"]  == "original_hash_123"
        assert alert["current_hash"] == "tampered_hash_456"
        assert alert["resolved"]     == False
        ok("Tamper alert saved to database")
    except Exception as e:
        fail("Save tamper alert", str(e))

    # Test 5.6 — Retrieve alerts
    try:
        alerts = get_all_alerts(resolved=False)
        assert len(alerts) >= 1
        ok(f"Tamper alerts retrieved ({len(alerts)} unresolved)")
    except Exception as e:
        fail("Retrieve tamper alerts", str(e))

    # Test 5.7 — Empty custody log for unknown ID
    try:
        log = get_custody_log("NONEXISTENT_ID_999")
        assert log == []
        ok("Empty list returned for unknown custody ID")
    except Exception as e:
        fail("Empty custody log for unknown ID", str(e))


# ══════════════════════════════════════════════════════
# TEST 6: API ENDPOINTS
# ══════════════════════════════════════════════════════
def test_api():
    section("TEST 6: Flask API Endpoints (app.py)")

    # Check server is running first
    try:
        resp = requests.get(f"{API_BASE}/health", timeout=5)
        if resp.status_code != 200:
            print(f"  {YELLOW}⚠️  Server not reachable. Start it with: python api/app.py{RESET}")
            return
        ok("Flask server is running")
    except requests.exceptions.ConnectionError:
        print(f"\n  {YELLOW}⚠️  SKIPPING API tests — server not running.{RESET}")
        print(f"  {YELLOW}   Start it with: python api/app.py{RESET}")
        return

    # Create a temp file to upload
    tmp = create_temp_file(b"API test evidence file content", ".txt")
    ev_id = f"APITEST_{int(datetime.utcnow().timestamp())}"

    # Test 6.1 — Health check
    try:
        resp = requests.get(f"{API_BASE}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] == True
        ok("GET /health returns 200")
    except Exception as e:
        fail("GET /health", str(e))

    # Test 6.2 — Upload file
    try:
        resp = requests.post(f"{API_BASE}/upload", json={
            "file_path":   tmp,
            "evidence_id": ev_id,
            "user":        "test_officer"
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["success"]     == True
        assert data["evidence_id"] == ev_id
        assert len(data["hash"])   == 64
        assert data["status"]      == "pending"
        ok("POST /upload returns 201 with hash and pending status")
    except Exception as e:
        fail("POST /upload", str(e))

    # Test 6.3 — Duplicate upload rejected
    try:
        resp = requests.post(f"{API_BASE}/upload", json={
            "file_path": tmp, "evidence_id": ev_id, "user": "test"
        })
        assert resp.status_code == 409
        ok("POST /upload returns 409 for duplicate evidence_id")
    except Exception as e:
        fail("POST /upload duplicate rejection", str(e))

    # Test 6.4 — Upload missing file
    try:
        resp = requests.post(f"{API_BASE}/upload", json={
            "file_path": "/nonexistent/file.pdf", "evidence_id": "MISSING001"
        })
        assert resp.status_code == 404
        ok("POST /upload returns 404 for missing file")
    except Exception as e:
        fail("POST /upload missing file 404", str(e))

    # Test 6.5 — Verify VERIFIED
    try:
        resp = requests.post(f"{API_BASE}/verify", json={
            "file_path": tmp, "evidence_id": ev_id, "user": "test_officer"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["integrity_status"] == "VERIFIED"
        assert data["alert_saved"]      == False
        ok("POST /verify returns VERIFIED for untampered file")
    except Exception as e:
        fail("POST /verify VERIFIED", str(e))

    # Test 6.6 — Verify TAMPERED
    try:
        tmp2   = create_temp_file(b"different file content!", ".txt")
        ev_id2 = f"APITEST2_{int(datetime.utcnow().timestamp())}"
        requests.post(f"{API_BASE}/upload", json={
            "file_path": tmp2, "evidence_id": ev_id2, "user": "test"
        })
        # Modify file after upload
        with open(tmp2, "wb") as f:
            f.write(b"this content has been tampered with!")
        resp = requests.post(f"{API_BASE}/verify", json={
            "file_path": tmp2, "evidence_id": ev_id2, "user": "test_officer"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["integrity_status"] == "TAMPERED"
        assert data["alert_saved"]      == True
        assert data["alert_id"]         is not None
        ok("POST /verify returns TAMPERED + alert saved for modified file")
        os.unlink(tmp2)
    except Exception as e:
        fail("POST /verify TAMPERED + alert", str(e))

    # Test 6.7 — Approve pending file (should fail — not verified yet... wait it is)
    try:
        resp = requests.post(f"{API_BASE}/approve", json={
            "evidence_id": ev_id,
            "approved_by": "senior_officer",
            "notes": "Confirmed authentic"
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["success"]     == True
        assert data["block_index"] >= 1
        ok("POST /approve adds verified file to blockchain")
    except Exception as e:
        fail("POST /approve", str(e))

    # Test 6.8 — Double approve rejected
    try:
        resp = requests.post(f"{API_BASE}/approve", json={
            "evidence_id": ev_id, "approved_by": "officer"
        })
        assert resp.status_code == 409
        ok("POST /approve returns 409 for already-approved file")
    except Exception as e:
        fail("POST /approve double approval rejection", str(e))

    # Test 6.9 — Blockchain endpoint
    try:
        resp = requests.get(f"{API_BASE}/blockchain")
        assert resp.status_code == 200
        data = resp.json()
        assert data["chain_valid"]  == True
        assert data["chain_length"] >= 2
        ok(f"GET /blockchain returns valid chain ({data['chain_length']} blocks)")
    except Exception as e:
        fail("GET /blockchain", str(e))

    # Test 6.10 — Alerts endpoint
    try:
        resp = requests.get(f"{API_BASE}/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert "alerts" in data
        assert data["alert_count"] >= 1
        ok(f"GET /alerts returns tamper alerts ({data['alert_count']} alerts)")
    except Exception as e:
        fail("GET /alerts", str(e))

    # Test 6.11 — Evidence list
    try:
        resp = requests.get(f"{API_BASE}/evidence")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1
        ok(f"GET /evidence returns all records ({data['count']} records)")
    except Exception as e:
        fail("GET /evidence", str(e))

    # Test 6.12 — Custody log
    try:
        resp = requests.get(f"{API_BASE}/custody/{ev_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["event_count"] >= 2
        ok(f"GET /custody/<id> returns audit trail ({data['event_count']} events)")
    except Exception as e:
        fail("GET /custody/<id>", str(e))

    # Test 6.13 — Filter evidence by type
    try:
        resp = requests.get(f"{API_BASE}/evidence?type=document")
        assert resp.status_code == 200
        ok("GET /evidence?type=document filter works")
    except Exception as e:
        fail("GET /evidence?type filter", str(e))

    os.unlink(tmp)


# ══════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════
def print_final_summary():
    total = passed + failed
    print(f"\n{BOLD}{'═'*55}{RESET}")
    print(f"{BOLD}  FINAL TEST RESULTS{RESET}")
    print(f"{BOLD}{'═'*55}{RESET}")
    print(f"  Total tests : {total}")
    print(f"  {GREEN}✅ Passed   : {passed}{RESET}")
    print(f"  {RED}❌ Failed   : {failed}{RESET}")
    print(f"  Score       : {int(passed/total*100) if total > 0 else 0}%")
    print(f"{BOLD}{'═'*55}{RESET}")
    if failed == 0:
        print(f"\n  {GREEN}{BOLD}ALL TESTS PASSED! Module is working correctly.{RESET}")
    else:
        print(f"\n  {YELLOW}{BOLD}{failed} test(s) failed. Check errors above.{RESET}")
    print()


# ══════════════════════════════════════════════════════
# RUN ALL TESTS
# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{BOLD}{'═'*55}{RESET}")
    print(f"{BOLD}  Cryptographic Integrity & Blockchain — Test Suite{RESET}")
    print(f"{BOLD}{'═'*55}{RESET}")
    print(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    test_hashing()
    test_blockchain()
    test_verify()
    test_database()
    test_custody()
    test_api()

    print_final_summary()
