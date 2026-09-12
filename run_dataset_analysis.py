"""
run_dataset_analysis.py
------------------------
Batch processor — run forensic analysis on a directory of evidence files.

Usage
-----
# Analyse all supported files in a directory
python run_dataset_analysis.py --dir path/to/evidence_folder

# Analyse with user attribution
python run_dataset_analysis.py --dir path/to/folder --user batch_processor

# Save summary to CSV
python run_dataset_analysis.py --dir path/to/folder --csv results.csv

# Filter by file type
python run_dataset_analysis.py --dir path/to/folder --type image

# Limit number of files
python run_dataset_analysis.py --dir path/to/folder --limit 10
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import configure_logging, SUPPORTED_EXTENSIONS
from orchestration.analysis_orchestrator import analyze_evidence
from orchestration.evidence_router import route_evidence


def collect_files(directory: Path, file_type_filter: str | None = None) -> list[Path]:
    """Collect all supported evidence files from a directory (non-recursive)."""
    files = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        decision = route_evidence(str(path))
        if not decision.is_supported:
            continue
        if file_type_filter and decision.file_type != file_type_filter:
            continue
        files.append(path)
    return files


def run_batch(
    directory: Path,
    submitted_by: str = "batch_processor",
    file_type_filter: str | None = None,
    limit: int | None = None,
    csv_path: Path | None = None,
) -> list[dict]:
    """
    Analyse all supported files in a directory.

    Returns
    -------
    list[dict]
        Summary of results for each file (not the full report).
    """
    files = collect_files(directory, file_type_filter)
    if limit:
        files = files[:limit]

    total = len(files)
    logger = logging.getLogger(__name__)
    logger.info("Batch analysis: %d files in %s", total, directory)

    summaries = []
    failed_count = 0

    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{total}] Analysing: {file_path.name}")
        t0 = time.perf_counter()
        try:
            report = analyze_evidence(str(file_path), submitted_by=submitted_by)
            report_dict = report.to_dict()
            elapsed = time.perf_counter() - t0

            summary = {
                "file_name":      file_path.name,
                "evidence_id":    report_dict.get("evidence_id"),
                "file_type":      report_dict.get("file", {}).get("file_type"),
                "sha256":         report_dict.get("file", {}).get("sha256", "")[:16] + "…",
                "overall_status": report_dict.get("overall_status"),
                "elapsed_s":      round(elapsed, 2),
                # Per-module verdicts
                "blockchain":     _verdict(report_dict, "blockchain"),
                "metadata_score": _meta_score(report_dict),
                "image_forgery":  _verdict(report_dict, "image_forgery"),
                "deepfake":       _verdict(report_dict, "deepfake"),
                "fake_news":      _verdict(report_dict, "fake_news"),
            }
            summaries.append(summary)
            print(f"    → {summary['overall_status']}  ({elapsed:.1f}s)")

        except Exception as exc:
            failed_count += 1
            elapsed = time.perf_counter() - t0
            summary = {
                "file_name":      file_path.name,
                "evidence_id":    "ERROR",
                "file_type":      "N/A",
                "sha256":         "N/A",
                "overall_status": "ERROR",
                "elapsed_s":      round(elapsed, 2),
                "error":          str(exc),
            }
            summaries.append(summary)
            logger.error("Failed to analyse %s: %s", file_path.name, exc)

    # ── Print final summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  BATCH COMPLETE")
    print(f"  Total files : {total}")
    print(f"  Succeeded   : {total - failed_count}")
    print(f"  Failed      : {failed_count}")
    print(f"{'='*60}")

    # ── Write CSV ──────────────────────────────────────────────────────────────
    if csv_path and summaries:
        _write_csv(summaries, csv_path)
        print(f"\n  Summary CSV saved: {csv_path}")

    return summaries


def _verdict(report: dict, module: str) -> str:
    """Extract a short verdict string from a module result."""
    data = report.get(module)
    if data is None:
        return "N/A"
    status = data.get("status", "N/A")
    if status == "skipped":
        return "skipped"
    if status == "failed":
        return f"failed: {data.get('error', '')[:40]}"
    result = data.get("result", {})
    return (
        result.get("prediction")
        or result.get("final_decision")
        or result.get("integrity_status")
        or status
    )


def _meta_score(report: dict) -> str:
    data = report.get("metadata")
    if not data or data.get("status") != "success":
        return "N/A"
    return str(data.get("result", {}).get("metadata_score", "N/A"))


def _write_csv(summaries: list[dict], csv_path: Path) -> None:
    if not summaries:
        return
    fieldnames = list(summaries[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)


def main():
    configure_logging("INFO")

    parser = argparse.ArgumentParser(
        prog="run_dataset_analysis.py",
        description="Batch forensic analysis — process a folder of evidence files",
    )
    parser.add_argument("--dir",   "-d", required=True, help="Directory of evidence files")
    parser.add_argument("--user",  "-u", default="batch_processor")
    parser.add_argument("--type",  "-t", choices=["image", "video", "document"],
                        help="Only analyse files of this type")
    parser.add_argument("--limit", "-n", type=int, help="Process at most N files")
    parser.add_argument("--csv",   "-c", help="Save summary table to a CSV file")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()
    if args.verbose:
        configure_logging("DEBUG")

    directory = Path(args.dir).resolve()
    if not directory.is_dir():
        print(f"[ERROR] Not a directory: {directory}", file=sys.stderr)
        sys.exit(1)

    csv_path = Path(args.csv) if args.csv else None

    run_batch(
        directory=directory,
        submitted_by=args.user,
        file_type_filter=args.type,
        limit=args.limit,
        csv_path=csv_path,
    )


if __name__ == "__main__":
    main()
