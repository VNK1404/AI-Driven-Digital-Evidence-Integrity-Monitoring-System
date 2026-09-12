#!/usr/bin/env python3
"""
=============================================================================
  run_full_metadata_forensics.py
  Production-Ready Dataset Processing System for Digital Evidence Forensics
=============================================================================

Recursively scans ``datasets/`` and runs every file through the full
metadata forensic pipeline:

    metadata extraction → rule checks → scoring → timeline → anomaly detection

Results are appended in batches to ``forensics_results_full.csv``.
If interrupted (Ctrl+C), simply re-run — already-processed files are skipped.

Usage
-----
    cd metadata_forensics
    python run_full_metadata_forensics.py
"""

import os
import sys
import csv
import time
import logging
from pathlib import Path
from datetime import datetime

from tqdm import tqdm

from pipeline.metadata_pipeline import run_metadata_pipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forensics_results_full.csv")
BATCH_SIZE = 50          # flush to disk every N files
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forensics_processing.log")

# Supported file extensions mapped to file types
EXTENSION_MAP = {
    # Images
    ".jpg": "image", ".jpeg": "image", ".png": "image",
    ".bmp": "image", ".tif": "image", ".tiff": "image",
    # Videos
    ".mp4": "video", ".avi": "video", ".mov": "video", ".mkv": "video",
    # Audio
    ".wav": "audio", ".mp3": "audio", ".flac": "audio",
    # Documents
    ".pdf": "document",
}

CSV_COLUMNS = [
    "dataset", "category", "file_path", "file_type",
    "metadata_score", "flags", "timeline_flags", "anomaly",
]

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataset → label mapping
# ---------------------------------------------------------------------------

# Order matters: more specific patterns first.
# Each tuple: (path_fragment_lower, dataset_label, category_label)
LABEL_RULES = [
    # CASIA2
    ("casia2/au",                    "CASIA2",      "authentic_image"),
    ("casia2\\au",                   "CASIA2",      "authentic_image"),
    ("casia2/tp",                    "CASIA2",      "tampered_image"),
    ("casia2\\tp",                   "CASIA2",      "tampered_image"),
    # Columbia
    ("columbia/4cam_auth",           "Columbia",    "authentic_image"),
    ("columbia\\4cam_auth",          "Columbia",    "authentic_image"),
    ("columbia/4cam_splc",           "Columbia",    "spliced_image"),
    ("columbia\\4cam_splc",          "Columbia",    "spliced_image"),
    # FF++
    ("ff++/real",                    "FF++",        "real_video"),
    ("ff++\\real",                   "FF++",        "real_video"),
    ("ff++/fake",                    "FF++",        "deepfake_video"),
    ("ff++\\fake",                   "FF++",        "deepfake_video"),
    # LJSpeech
    ("ljspeech",                     "LJSpeech-1.1","audio"),
    # CompanyDocuments
    ("companydocuments",             "CompanyDocuments", "document"),
]


def classify_file(file_path: str):
    """Return (dataset, category) for a file based on its path."""
    path_lower = file_path.lower()
    for fragment, dataset, category in LABEL_RULES:
        if fragment in path_lower:
            return dataset, category
    return "Unknown", "unknown"


def detect_file_type(file_path: str):
    """Return the file type string based on extension, or None if unsupported."""
    ext = os.path.splitext(file_path)[1].lower()
    return EXTENSION_MAP.get(ext)

# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------


def load_processed_files(csv_path: str):
    """Return a set of file paths already present in the CSV."""
    processed = set()
    if os.path.isfile(csv_path):
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    processed.add(row.get("file_path", ""))
        except Exception as exc:
            logger.warning(f"Could not read existing CSV for resume: {exc}")
    return processed

# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------


def discover_files(root_dir: str):
    """Recursively discover all supported files under *root_dir*.

    Returns a list of (absolute_path, file_type, dataset, category) tuples.
    """
    file_list = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            full_path = os.path.join(dirpath, fname)
            ftype = detect_file_type(full_path)
            if ftype is None:
                continue  # skip unsupported extensions
            dataset, category = classify_file(full_path)
            file_list.append((full_path, ftype, dataset, category))
    return file_list

# ---------------------------------------------------------------------------
# Batch writer
# ---------------------------------------------------------------------------


class BatchCSVWriter:
    """Buffers rows and flushes them to disk every *batch_size* rows."""

    def __init__(self, csv_path: str, columns: list, batch_size: int = 50):
        self.csv_path = csv_path
        self.columns = columns
        self.batch_size = batch_size
        self._buffer = []
        self._file_exists = os.path.isfile(csv_path)

    def add(self, row: dict):
        self._buffer.append(row)
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        write_header = not self._file_exists
        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.columns)
                if write_header:
                    writer.writeheader()
                    self._file_exists = True
                writer.writerows(self._buffer)
        except Exception as exc:
            logger.error(f"Failed to write batch to CSV: {exc}")
        self._buffer.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.flush()

# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------


def process_all():
    """Entry point — discover, process, and record results."""
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("  Metadata Forensics — Full Dataset Processing")
    logger.info("=" * 70)
    logger.info(f"Datasets directory : {DATASETS_DIR}")
    logger.info(f"Output CSV         : {OUTPUT_CSV}")
    logger.info(f"Batch size         : {BATCH_SIZE}")
    logger.info("")

    # 1. Discover files -------------------------------------------------------
    logger.info("Scanning for supported files …")
    all_files = discover_files(DATASETS_DIR)
    total_files = len(all_files)
    logger.info(f"Found {total_files} supported files.\n")

    if total_files == 0:
        logger.warning("No supported files found. Exiting.")
        return

    # 2. Resume support -------------------------------------------------------
    processed = load_processed_files(OUTPUT_CSV)
    skipped_count = 0
    pending_files = []
    for entry in all_files:
        if entry[0] in processed:
            skipped_count += 1
        else:
            pending_files.append(entry)

    if skipped_count:
        logger.info(f"Resuming — skipping {skipped_count} already-processed files.")
    logger.info(f"Files to process: {len(pending_files)}\n")

    if not pending_files:
        logger.info("All files already processed. Nothing to do.")
        return

    # 3. Group by dataset/category for informative progress --------------------
    from collections import OrderedDict
    groups = OrderedDict()
    for fpath, ftype, dataset, category in pending_files:
        key = (dataset, category)
        groups.setdefault(key, []).append((fpath, ftype))

    # 4. Process ---------------------------------------------------------------
    success_count = 0
    error_count = 0

    with BatchCSVWriter(OUTPUT_CSV, CSV_COLUMNS, BATCH_SIZE) as writer:
        for (dataset, category), files in groups.items():
            logger.info(f"Processing {dataset} {category} … ({len(files)} files)")

            for fpath, ftype in tqdm(files, desc=f"{dataset}/{category}", unit="file", leave=True):
                try:
                    result = run_metadata_pipeline(fpath, file_type=ftype)

                    row = {
                        "dataset": dataset,
                        "category": category,
                        "file_path": fpath,
                        "file_type": ftype,
                        "metadata_score": result.get("score", -1),
                        "flags": "; ".join(result.get("flags", [])),
                        "timeline_flags": "; ".join(result.get("timeline_flags", [])),
                        "anomaly": result.get("anomaly", False),
                    }
                    writer.add(row)
                    success_count += 1

                except KeyboardInterrupt:
                    logger.info("\nInterrupted by user — flushing buffered results …")
                    writer.flush()
                    logger.info("Progress saved. You can resume by re-running this script.")
                    sys.exit(0)

                except Exception as exc:
                    error_count += 1
                    logger.debug(f"Error processing {fpath}: {exc}")

    # 5. Summary ---------------------------------------------------------------
    elapsed = time.time() - start_time
    logger.info("")
    logger.info("=" * 70)
    logger.info("  Processing Complete")
    logger.info("=" * 70)
    logger.info(f"Total files found       : {total_files}")
    logger.info(f"Already processed (skip): {skipped_count}")
    logger.info(f"Successfully processed  : {success_count}")
    logger.info(f"Errors                  : {error_count}")
    logger.info(f"Time elapsed            : {elapsed:.1f}s")
    logger.info(f"Results saved to        : {OUTPUT_CSV}")
    logger.info("")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    process_all()
