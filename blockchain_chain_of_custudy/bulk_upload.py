"""
bulk_upload.py
--------------
Bulk upload script for the Evidence Integrity System.
Handles ALL file types: images, videos, audio, documents.

HOW IT WORKS:
  - Point it at a folder containing your files
  - It scans ALL subfolders automatically
  - Uploads every supported file to the API
  - Auto-verifies each file after upload
  - Saves a full log of results to CSV
  - Skips already uploaded files (safe to re-run)

FOLDER STRUCTURE EXPECTED (flexible — any structure works):
    your_folder/
    ├── images/
    │   ├── photo1.jpg
    │   └── scan.tif
    ├── videos/
    │   └── recording.mp4
    ├── audio/
    │   └── interview.mp3
    └── documents/
        └── report.pdf

USAGE:
    # Upload only
    python bulk_upload.py --folder "C:/Users/parth/evidence"

    # Upload + auto verify everything
    python bulk_upload.py --folder "C:/Users/parth/evidence" --verify

    # Upload + verify + auto approve verified files
    python bulk_upload.py --folder "C:/Users/parth/evidence" --verify --approve

    # Preview what will be uploaded (no actual upload)
    python bulk_upload.py --folder "C:/Users/parth/evidence" --dry-run

    # Faster with more threads
    python bulk_upload.py --folder "C:/Users/parth/evidence" --threads 10
"""

import os
import sys
import csv
import time
import argparse
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────
API_BASE    = "http://127.0.0.1:5001"
UPLOAD_URL  = f"{API_BASE}/upload"
VERIFY_URL  = f"{API_BASE}/verify"
APPROVE_URL = f"{API_BASE}/approve"
EVIDENCE_URL= f"{API_BASE}/evidence"

DEFAULT_THREADS = 5
DEFAULT_USER    = "bulk_upload_script"
LOG_FILE        = "bulk_upload_log.csv"
FAILED_FILE     = "failed_uploads.txt"

# All supported file types
SUPPORTED = {
    "image":    {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"},
    "video":    {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"},
    "audio":    {".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a"},
    "document": {".pdf", ".docx", ".doc", ".txt", ".xlsx", ".xls", ".pptx"},
}
ALL_EXTENSIONS = {ext for exts in SUPPORTED.values() for ext in exts}


# ── Thread-safe counter ───────────────────────────────────────────────────────
class Counter:
    def __init__(self):
        self._val  = 0
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self._val += 1
            return self._val

    @property
    def value(self):
        return self._val


# ── File utilities ────────────────────────────────────────────────────────────
def get_file_category(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    for category, exts in SUPPORTED.items():
        if ext in exts:
            return category
    return "unknown"


def scan_folder(folder_path: str) -> list[dict]:
    """
    Recursively scan a folder for all supported files.

    Returns a list of dicts: {file_path, file_name, category, evidence_id}
    Evidence IDs are auto-generated: IMG_00001, VID_00001, AUD_00001, DOC_00001
    """
    files = []
    counters = {"image": 0, "video": 0, "audio": 0, "document": 0}
    prefix_map = {"image": "IMG", "video": "VID", "audio": "AUD", "document": "DOC"}

    for root, _, filenames in os.walk(folder_path):
        for fname in sorted(filenames):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in ALL_EXTENSIONS:
                continue

            category = get_file_category(fname)
            counters[category] += 1
            prefix   = prefix_map[category]
            base     = os.path.splitext(fname)[0].replace(" ", "_")[:25]
            ev_id    = f"{prefix}_{counters[category]:05d}_{base}"

            files.append({
                "file_path":   os.path.abspath(os.path.join(root, fname)),
                "file_name":   fname,
                "category":    category,
                "evidence_id": ev_id,
            })

    return files


def fetch_uploaded_ids() -> set[str]:
    """Fetch all evidence IDs already in the database for resume support."""
    try:
        resp = requests.get(EVIDENCE_URL, timeout=10)
        if resp.status_code == 200:
            return {r["evidence_id"] for r in resp.json().get("evidence", [])}
    except Exception:
        pass
    return set()


# ── API calls ─────────────────────────────────────────────────────────────────
def do_upload(item: dict, user: str) -> dict:
    """Upload a single file to the API."""
    try:
        resp = requests.post(
            UPLOAD_URL,
            json={"file_path": item["file_path"], "evidence_id": item["evidence_id"], "user": user},
            timeout=60,
        )
        data = resp.json()
        if resp.status_code == 201:
            return {"step": "upload", "status": "SUCCESS", "data": data, "error": ""}
        elif resp.status_code == 409:
            return {"step": "upload", "status": "SKIPPED", "data": data, "error": "Already exists"}
        else:
            return {"step": "upload", "status": "FAILED",  "data": {}, "error": data.get("error", "Unknown")}
    except requests.exceptions.ConnectionError:
        return {"step": "upload", "status": "FAILED", "data": {}, "error": "Cannot connect to API. Is server running?"}
    except Exception as e:
        return {"step": "upload", "status": "FAILED", "data": {}, "error": str(e)}


def do_verify(item: dict, user: str) -> dict:
    """Verify a single file's integrity."""
    try:
        resp = requests.post(
            VERIFY_URL,
            json={"file_path": item["file_path"], "evidence_id": item["evidence_id"], "user": user},
            timeout=60,
        )
        data = resp.json()
        if resp.status_code == 200:
            return {
                "step":   "verify",
                "status": data.get("integrity_status", "ERROR"),
                "data":   data,
                "error":  "",
            }
        else:
            return {"step": "verify", "status": "ERROR", "data": {}, "error": data.get("error", "Unknown")}
    except Exception as e:
        return {"step": "verify", "status": "ERROR", "data": {}, "error": str(e)}


def do_approve(item: dict, user: str) -> dict:
    """Approve a verified file for the blockchain."""
    try:
        resp = requests.post(
            APPROVE_URL,
            json={"evidence_id": item["evidence_id"], "approved_by": user, "notes": "Bulk approved"},
            timeout=30,
        )
        data = resp.json()
        if resp.status_code == 201:
            return {"step": "approve", "status": "APPROVED", "data": data, "error": ""}
        else:
            return {"step": "approve", "status": "FAILED", "data": {}, "error": data.get("error", "Unknown")}
    except Exception as e:
        return {"step": "approve", "status": "FAILED", "data": {}, "error": str(e)}


# ── Progress display ──────────────────────────────────────────────────────────
def progress_bar(done: int, total: int, start_time: float, fname: str, status: str):
    elapsed  = time.time() - start_time
    rate     = done / elapsed if elapsed > 0 else 0
    eta      = int((total - done) / rate) if rate > 0 else 0
    bar_len  = 20
    filled   = int(bar_len * done / total)
    bar      = "█" * filled + "░" * (bar_len - filled)
    pct      = int(100 * done / total)
    icon     = "✅" if "SUCCESS" in status or "VERIFIED" in status or "APPROVED" in status else \
               "⏭" if "SKIP" in status else "❌"
    print(
        f"\r  [{bar}] {pct:3d}%  {done:,}/{total:,}  ETA:{eta}s  "
        f"{icon} {fname[:28]:<28}",
        end="", flush=True,
    )


# ── Summary ───────────────────────────────────────────────────────────────────
def print_summary(results: list[dict], elapsed: float, do_verify_flag: bool, do_approve_flag: bool):
    uploaded = sum(1 for r in results if r["upload"] == "SUCCESS")
    skipped  = sum(1 for r in results if r["upload"] == "SKIPPED")
    failed   = sum(1 for r in results if r["upload"] == "FAILED")

    by_type = {}
    for r in results:
        t = r["category"]
        by_type[t] = by_type.get(t, 0) + (1 if r["upload"] == "SUCCESS" else 0)

    print("\n\n" + "=" * 58)
    print("  BULK UPLOAD COMPLETE")
    print("=" * 58)
    print(f"  Total files    : {len(results):,}")
    print(f"  ✅ Uploaded    : {uploaded:,}")
    print(f"  ⏭  Skipped     : {skipped:,}  (already in DB)")
    print(f"  ❌ Failed      : {failed:,}")
    print(f"  ⏱  Time taken  : {elapsed:.1f}s")
    print()
    print("  By file type:")
    for t, count in by_type.items():
        print(f"    {t:<12}: {count:,}")

    if do_verify_flag:
        verified  = sum(1 for r in results if r.get("verify") == "VERIFIED")
        tampered  = sum(1 for r in results if r.get("verify") == "TAMPERED")
        print(f"\n  Integrity check:")
        print(f"    ✅ Verified  : {verified:,}")
        print(f"    ❌ Tampered  : {tampered:,}")

    if do_approve_flag:
        approved = sum(1 for r in results if r.get("approve") == "APPROVED")
        print(f"\n  Blockchain:")
        print(f"    🔗 Approved  : {approved:,}")

    print(f"\n  📄 Full log    : {LOG_FILE}")
    if failed:
        print(f"  ⚠️  Failed list : {FAILED_FILE}")
    print("=" * 58)


def save_logs(results: list[dict]):
    """Save full results to CSV."""
    fields = ["evidence_id", "file_name", "category", "file_path",
              "upload", "verify", "approve", "error"]
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)

    failed = [r for r in results if r["upload"] == "FAILED"]
    if failed:
        with open(FAILED_FILE, "w", encoding="utf-8") as f:
            for r in failed:
                f.write(f"{r['evidence_id']} | {r['file_name']} | {r['error']}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def bulk_upload(folder: str, threads: int, user: str,
                do_verify_flag: bool, do_approve_flag: bool, dry_run: bool):

    # ── Scan folder ───────────────────────────────────────────────────────
    print(f"\n🔍 Scanning: {folder}")
    all_files = scan_folder(folder)

    if not all_files:
        print("[ERROR] No supported files found in that folder.")
        print(f"Supported types: {', '.join(sorted(ALL_EXTENSIONS))}")
        sys.exit(1)

    # Count by category
    cats = {}
    for f in all_files:
        cats[f["category"]] = cats.get(f["category"], 0) + 1

    print("=" * 58)
    print("  Evidence Integrity System — Bulk Upload")
    print("=" * 58)
    print(f"  Folder   : {folder}")
    for cat, count in cats.items():
        icon = {"image":"🖼", "video":"🎬", "audio":"🎵", "document":"📄"}.get(cat, "📁")
        print(f"  {icon} {cat:<10}: {count:,} files")
    print(f"  Total    : {len(all_files):,} files")
    print(f"  Threads  : {threads}")
    print(f"  Verify   : {'YES' if do_verify_flag else 'NO'}")
    print(f"  Approve  : {'YES — verified files go to blockchain' if do_approve_flag else 'NO'}")
    print("=" * 58)

    # ── Dry run ───────────────────────────────────────────────────────────
    if dry_run:
        print(f"\n[DRY RUN] First 15 files that would be uploaded:\n")
        for item in all_files[:15]:
            print(f"  [{item['category'][:3].upper()}]  {item['evidence_id']:<40}  {item['file_name']}")
        if len(all_files) > 15:
            print(f"  ... and {len(all_files) - 15} more.")
        print("\nRun without --dry-run to start uploading.")
        return

    # ── Resume: skip already uploaded ────────────────────────────────────
    print("\n⏩ Checking for already uploaded files...")
    uploaded_ids = fetch_uploaded_ids()
    to_process   = [f for f in all_files if f["evidence_id"] not in uploaded_ids]
    skipped_pre  = len(all_files) - len(to_process)

    if skipped_pre:
        print(f"  Skipping {skipped_pre:,} already uploaded files.")
    print(f"  Processing {len(to_process):,} new files with {threads} threads...\n")

    if not to_process:
        print("✅ All files already uploaded!")
        return

    # ── Upload ────────────────────────────────────────────────────────────
    results      = []
    results_lock = threading.Lock()
    counter      = Counter()
    total        = len(to_process)
    start_time   = time.time()

    def process(item):
        # Step 1: Upload
        up = do_upload(item, user)

        row = {
            "evidence_id": item["evidence_id"],
            "file_name":   item["file_name"],
            "category":    item["category"],
            "file_path":   item["file_path"],
            "upload":      up["status"],
            "verify":      "",
            "approve":     "",
            "error":       up["error"],
        }

        # Step 2: Verify (if flag set and upload succeeded)
        if do_verify_flag and up["status"] == "SUCCESS":
            vr = do_verify(item, user)
            row["verify"] = vr["status"]
            if vr["status"] != "VERIFIED":
                row["error"] = vr.get("error", "Tampered")

            # Step 3: Approve (if flag set and verified)
            if do_approve_flag and vr["status"] == "VERIFIED":
                ap = do_approve(item, user)
                row["approve"] = ap["status"]

        count = counter.increment()
        with results_lock:
            results.append(row)

        status = row["approve"] or row["verify"] or row["upload"]
        progress_bar(count, total, start_time, item["file_name"], status)

    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = [executor.submit(process, item) for item in to_process]
        for future in as_completed(futures):
            future.result()

    # Add pre-skipped to results for log
    for item in all_files:
        if item["evidence_id"] in uploaded_ids:
            results.append({
                "evidence_id": item["evidence_id"],
                "file_name":   item["file_name"],
                "category":    item["category"],
                "file_path":   item["file_path"],
                "upload":      "SKIPPED",
                "verify":      "",
                "approve":     "",
                "error":       "",
            })

    elapsed = time.time() - start_time
    save_logs(results)
    print_summary(results, elapsed, do_verify_flag, do_approve_flag)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bulk upload any files to Evidence Integrity System"
    )
    parser.add_argument("--folder",   required=True, help='Path to your evidence folder')
    parser.add_argument("--threads",  type=int, default=DEFAULT_THREADS, help="Parallel threads (default: 5)")
    parser.add_argument("--user",     default=DEFAULT_USER, help="Username for custody logs")
    parser.add_argument("--verify",   action="store_true", help="Auto-verify each file after upload")
    parser.add_argument("--approve",  action="store_true", help="Auto-approve verified files to blockchain")
    parser.add_argument("--dry-run",  action="store_true", help="Preview only, no actual upload")

    args = parser.parse_args()

    # --approve requires --verify
    if args.approve and not args.verify:
        print("[ERROR] --approve requires --verify. Use: --verify --approve")
        sys.exit(1)

    bulk_upload(
        folder          = args.folder,
        threads         = args.threads,
        user            = args.user,
        do_verify_flag  = args.verify,
        do_approve_flag = args.approve,
        dry_run         = args.dry_run,
    )
