#!/usr/bin/env python3
"""
=============================================================================
  test_pipeline.py - Comprehensive Test Suite for Metadata Forensics Module
=============================================================================
Tests every component: extractors, features, rules, anomaly, scoring, pipeline.
Uses real files from the datasets/ folder when available.

Usage:
    cd metadata_forensics
    python test_pipeline.py
"""

import io
import os
import sys
import glob
import traceback

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS = 0
FAIL = 0
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS = os.path.join(BASE_DIR, "datasets")


def log_pass(name):
    global PASS
    PASS += 1
    print(f"  [PASS] {name}")


def log_fail(name, err=""):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {name} -- {err}")


def find_sample(pattern_list):
    """Find a real sample file from datasets/."""
    for pattern in pattern_list:
        matches = glob.glob(os.path.join(DATASETS, "**", pattern), recursive=True)
        if matches:
            return matches[0]
    return None


# ---------------------------------------------------------------------------
# Test 1: Individual Extractors
# ---------------------------------------------------------------------------
def test_image_extractor():
    print("\n── Test 1a: Image Extractor ──")
    from extraction.image_extractor import extract_image_metadata

    sample = find_sample(["*.jpg", "*.jpeg", "*.png", "*.tif", "*.bmp"])
    if not sample:
        log_fail("image_extractor", "No image files found in datasets/")
        return

    meta = extract_image_metadata(sample)
    if isinstance(meta, dict):
        log_pass(f"image_extractor returned dict ({len(meta)} keys) for {os.path.basename(sample)}")
    else:
        log_fail("image_extractor", f"Expected dict, got {type(meta)}")

    # Test corrupt path
    meta2 = extract_image_metadata("nonexistent_file.jpg")
    if isinstance(meta2, dict) and "extraction_error" in meta2:
        log_pass("image_extractor handles missing file")
    else:
        log_fail("image_extractor missing file handling")


def test_video_extractor():
    print("\n── Test 1b: Video Extractor ──")
    from extraction.video_extractor import extract_video_metadata

    sample = find_sample(["*.mp4", "*.avi", "*.mov", "*.mkv"])
    if not sample:
        print("  [SKIP] No video files found in datasets/")
        return

    meta = extract_video_metadata(sample)
    if isinstance(meta, dict):
        log_pass(f"video_extractor returned dict ({len(meta)} keys)")
    else:
        log_fail("video_extractor", f"Expected dict, got {type(meta)}")


def test_audio_extractor():
    print("\n── Test 1c: Audio Extractor ──")
    from extraction.audio_extractor import extract_audio_metadata

    sample = find_sample(["*.wav", "*.mp3", "*.flac"])
    if not sample:
        print("  [SKIP] No audio files found in datasets/")
        return

    meta = extract_audio_metadata(sample)
    if isinstance(meta, dict):
        log_pass(f"audio_extractor returned dict ({len(meta)} keys)")
        if "duration" in meta:
            log_pass(f"  duration = {meta['duration']:.2f}s")
        if "sample_rate" in meta:
            log_pass(f"  sample_rate = {meta['sample_rate']}")
    else:
        log_fail("audio_extractor", f"Expected dict, got {type(meta)}")


def test_document_extractor():
    print("\n── Test 1d: Document Extractor ──")
    from extraction.document_extractor import extract_document_metadata

    sample = find_sample(["*.pdf"])
    if not sample:
        print("  [SKIP] No PDF files found in datasets/")
        return

    meta = extract_document_metadata(sample)
    if isinstance(meta, dict):
        log_pass(f"document_extractor returned dict ({len(meta)} keys)")
        if "page_count" in meta:
            log_pass(f"  page_count = {meta['page_count']}")
    else:
        log_fail("document_extractor", f"Expected dict, got {type(meta)}")


# ---------------------------------------------------------------------------
# Test 2: Unified extract_metadata()
# ---------------------------------------------------------------------------
def test_extract_metadata():
    print("\n── Test 2: Unified extract_metadata() ──")
    from extraction import extract_metadata

    for ext_list, expected_type in [
        (["*.jpg", "*.jpeg", "*.png"], "image"),
        (["*.mp4", "*.avi"], "video"),
        (["*.wav", "*.mp3"], "audio"),
        (["*.pdf"], "document"),
    ]:
        sample = find_sample(ext_list)
        if not sample:
            print(f"  [SKIP] No {expected_type} files found")
            continue

        result = extract_metadata(sample)
        if result["file_type"] == expected_type:
            log_pass(f"extract_metadata auto-detected '{expected_type}' for {os.path.basename(sample)}")
        else:
            log_fail(f"extract_metadata type detection", f"Expected '{expected_type}', got '{result['file_type']}'")

        if isinstance(result["metadata"], dict):
            log_pass(f"  metadata is dict with {len(result['metadata'])} keys")
        else:
            log_fail("  metadata type", f"Expected dict, got {type(result['metadata'])}")

    # Unsupported extension
    result = extract_metadata("file.xyz")
    if result["file_type"] == "unknown":
        log_pass("extract_metadata handles unsupported extensions")
    else:
        log_fail("unsupported extension handling")


# ---------------------------------------------------------------------------
# Test 3: Feature Engineering
# ---------------------------------------------------------------------------
def test_features():
    print("\n── Test 3: metadata_to_features() ──")
    from analysis.features import metadata_to_features, features_to_vector

    # Rich metadata (should detect camera, software, etc.)
    rich_meta = {
        "Make": "Canon",
        "Model": "EOS R5",
        "Software": "Adobe Photoshop",
        "GPSInfo": {"lat": 40.7, "lon": -74.0},
        "DateTime": "2025:01:15 10:30:00",
        "ExposureTime": "1/250",
        "FNumber": 2.8,
    }

    feats = metadata_to_features(rich_meta, "image")
    assert feats["camera_present"] == 1, "camera should be detected"
    assert feats["editing_software_present"] == 1, "photoshop should be detected"
    assert feats["gps_present"] == 1, "GPS should be detected"
    assert feats["metadata_length"] == 7, f"Expected 7 keys, got {feats['metadata_length']}"
    assert feats["timestamp_present"] == 1, "DateTime should be detected"
    log_pass("metadata_to_features — rich metadata")

    # Empty metadata
    feats2 = metadata_to_features({}, "image")
    assert feats2["camera_present"] == 0
    assert feats2["metadata_length"] == 0
    log_pass("metadata_to_features — empty metadata")

    # features_to_vector
    vec = features_to_vector(feats)
    assert len(vec) == 7, f"vector should have 7 elements, got {len(vec)}"
    assert all(isinstance(v, (int, float)) for v in vec)
    log_pass("features_to_vector — correct shape and types")


# ---------------------------------------------------------------------------
# Test 4: Rule-Based Analysis
# ---------------------------------------------------------------------------
def test_rules():
    print("\n── Test 4: metadata_rules() ──")
    from analysis.rules import metadata_rules

    # Image with Photoshop and no camera
    meta = {"Software": "Adobe Photoshop CC 2024", "ExposureTime": "1/125"}
    flags = metadata_rules(meta, file_type="image")
    assert "editing_software_detected" in flags, "'editing_software_detected' expected"
    assert "camera_model_missing" in flags, "'camera_model_missing' expected"
    log_pass(f"image rules — {len(flags)} flags: {flags}")

    # Empty metadata
    flags2 = metadata_rules({}, file_type="image")
    assert "empty_metadata" in flags2
    log_pass(f"empty metadata flagged — {flags2}")

    # Audio rules
    audio_meta = {"duration": 0, "sample_rate": 4000, "channels": 1}
    flags3 = metadata_rules(audio_meta, file_type="audio")
    assert "low_sample_rate" in flags3
    assert "zero_or_negative_duration" in flags3
    log_pass(f"audio rules — {flags3}")


# ---------------------------------------------------------------------------
# Test 5: Anomaly Detection
# ---------------------------------------------------------------------------
def test_anomaly():
    print("\n── Test 5: Anomaly Detection ──")
    from analysis.anomaly_model import detect_anomaly, train_anomaly_model

    # Normal image features (camera present, no editing, some GPS, lots of keys)
    normal_features = [1, 0, 1, 20, 1, 1, 0]
    result = detect_anomaly(None, normal_features, file_type="image")
    log_pass(f"zero-training detection on normal image → anomaly={result}")

    # Suspicious features (no camera, editing present, no metadata)
    suspicious_features = [0, 1, 0, 1, 0, 0, 1]
    result2 = detect_anomaly(None, suspicious_features, file_type="image")
    log_pass(f"zero-training detection on suspicious image → anomaly={result2}")

    # Train IsolationForest with batch data
    train_data = [
        [1, 0, 1, 22, 1, 1, 0],  # normal
        [1, 0, 0, 18, 1, 1, 0],  # normal
        [1, 0, 1, 25, 1, 1, 0],  # normal
        [1, 0, 0, 20, 1, 1, 0],  # normal
        [1, 0, 1, 15, 1, 1, 0],  # normal
        [0, 1, 0, 2, 0, 0, 1],   # anomaly
    ]
    model = train_anomaly_model(train_data)
    result3 = detect_anomaly(model, [0, 1, 0, 1, 0, 0, 1], file_type="image")
    log_pass(f"IsolationForest detection on suspicious → anomaly={result3}")

    result4 = detect_anomaly(model, [1, 0, 1, 20, 1, 1, 0], file_type="image")
    log_pass(f"IsolationForest detection on normal → anomaly={result4}")


# ---------------------------------------------------------------------------
# Test 6: Scoring System
# ---------------------------------------------------------------------------
def test_scoring():
    print("\n── Test 6: compute_metadata_score() ──")
    from analysis.scoring import compute_metadata_score

    # ── Baseline ────────────────────────────────────────────────────────────
    # No flags, no anomaly → 100
    assert compute_metadata_score([], False) == 100
    log_pass("no flags, no anomaly → 100")

    # No flags + anomaly → 80  (−20 anomaly penalty)
    assert compute_metadata_score([], True) == 80
    log_pass("no flags + anomaly → 80")

    # ── INFO flags (0 pts each) ─────────────────────────────────────────────
    # GPS + software tag are informational — no deduction
    assert compute_metadata_score(["gps_data_present", "software_tag_present"], False) == 100
    log_pass("INFO flags (gps + software_tag) → 100 (no penalty)")

    # ── MEDIUM flags (−10 pts each) ─────────────────────────────────────────
    # 2 MEDIUM flags, no anomaly → 80
    assert compute_metadata_score(["camera_model_missing", "camera_make_missing"], False) == 80
    log_pass("2 MEDIUM flags (camera missing) → 80")

    # 1 MEDIUM flag + anomaly → 70  (−10 − 20)
    assert compute_metadata_score(["camera_model_missing"], True) == 70
    log_pass("1 MEDIUM flag + anomaly → 70")

    # ── HIGH flags (−25 pts each) ────────────────────────────────────────────
    # 1 HIGH flag, no anomaly → 75
    assert compute_metadata_score(["editing_software_detected"], False) == 75
    log_pass("1 HIGH flag (editing software) → 75")

    # 1 HIGH flag + anomaly → 55  (−25 − 20)
    assert compute_metadata_score(["editing_software_detected"], True) == 55
    log_pass("1 HIGH flag + anomaly → 55")

    # 4 HIGH flags + anomaly → 0 clamped  (4×−25 − 20 = −120)
    assert compute_metadata_score(
        ["editing_software_detected", "extraction_failed", "empty_metadata", "no_extraction_tool_available"],
        True
    ) == 0
    log_pass("4 HIGH flags + anomaly → 0 (clamped)")

    # ── Unknown flags default to MEDIUM ─────────────────────────────────────
    # Unknown flags fall back to MEDIUM (−10 each)
    assert compute_metadata_score(["unknown_flag_x", "unknown_flag_y"], False) == 80
    log_pass("2 unknown flags → MEDIUM fallback → 80")

    # ── Mixed severity ──────────────────────────────────────────────────────
    # 1 HIGH + 1 MEDIUM + 1 INFO + anomaly → 100 − 25 − 10 − 0 − 20 = 45
    assert compute_metadata_score(
        ["editing_software_detected", "camera_model_missing", "gps_data_present"],
        True
    ) == 45
    log_pass("mixed severity (HIGH + MEDIUM + INFO) + anomaly → 45")


# ---------------------------------------------------------------------------
# Test 7: Full Pipeline (end-to-end)
# ---------------------------------------------------------------------------
def test_pipeline():
    print("\n── Test 7: run_metadata_pipeline() (end-to-end) ──")
    from pipeline.metadata_pipeline import run_metadata_pipeline

    for ext_list, ftype in [
        (["*.jpg", "*.jpeg", "*.png"], "image"),
        (["*.wav", "*.mp3"], "audio"),
        (["*.pdf"], "document"),
    ]:
        sample = find_sample(ext_list)
        if not sample:
            print(f"  [SKIP] No {ftype} files found")
            continue

        result = run_metadata_pipeline(sample, model=None, file_type=ftype)

        # Validate output structure
        assert "metadata_score" in result, "missing 'metadata_score'"
        assert "flags" in result, "missing 'flags'"
        assert "anomaly" in result, "missing 'anomaly'"
        assert isinstance(result["metadata_score"], int), "score should be int"
        assert isinstance(result["flags"], list), "flags should be list"
        assert isinstance(result["anomaly"], bool), "anomaly should be bool"
        assert 0 <= result["metadata_score"] <= 100, f"score {result['metadata_score']} out of range"

        log_pass(
            f"pipeline({ftype}) → score={result['metadata_score']}, "
            f"flags={len(result['flags'])}, anomaly={result['anomaly']}"
        )

    # Auto file-type detection
    sample = find_sample(["*.jpg", "*.jpeg", "*.png"])
    if sample:
        result = run_metadata_pipeline(sample, model=None, file_type=None)
        assert "metadata_score" in result
        log_pass(f"pipeline(auto-detect) → score={result['metadata_score']}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  Metadata Forensics Module — Comprehensive Test Suite")
    print("=" * 70)

    tests = [
        test_image_extractor,
        test_video_extractor,
        test_audio_extractor,
        test_document_extractor,
        test_extract_metadata,
        test_features,
        test_rules,
        test_anomaly,
        test_scoring,
        test_pipeline,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as exc:
            log_fail(test_fn.__name__, str(exc))
            traceback.print_exc()

    print("\n" + "=" * 70)
    print(f"  Results: {PASS} passed, {FAIL} failed")
    print("=" * 70)

    if FAIL > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
