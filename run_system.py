"""
run_system.py
--------------
CLI entry point for the AI-Driven Digital Evidence Integrity Monitoring System.

Usage
-----
# Analyse a single file
python run_system.py --file path/to/evidence.jpg

# Analyse a single file with user attribution
python run_system.py --file path/to/evidence.mp4 --user investigator_alice

# Analyse a file and print full JSON to stdout
python run_system.py --file path/to/evidence.pdf --json

# Start the unified API server (port 8080)
python run_system.py --serve

# Start the API server on a custom port
python run_system.py --serve --port 9090
"""

import argparse
import json
import sys
import logging
from pathlib import Path

# ── Add project root to sys.path so imports resolve ───────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import configure_logging, API_HOST, API_PORT, API_DEBUG


def cmd_analyze(args: argparse.Namespace) -> int:
    """Run a single-file forensic analysis."""
    configure_logging("DEBUG" if args.verbose else "INFO")
    logger = logging.getLogger(__name__)

    from orchestration.analysis_orchestrator import analyze_evidence

    file_path = Path(args.file).resolve()
    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}", file=sys.stderr)
        return 1

    print(f"\n{'='*60}")
    print(f"  AI-DRIVEN FORENSIC ANALYSIS SYSTEM")
    print(f"{'='*60}")
    print(f"  File    : {file_path.name}")
    print(f"  Size    : {file_path.stat().st_size:,} bytes")
    print(f"  User    : {args.user}")
    print(f"{'='*60}\n")

    try:
        report = analyze_evidence(str(file_path), submitted_by=args.user)
        report_dict = report.to_dict()
        eid = report_dict.get("evidence_id", "N/A")

        if args.json:
            print(json.dumps(report_dict, indent=2, default=str))
        else:
            _print_summary(report_dict)

        print(f"\n  Report saved: reports/{eid}.json")
        return 0

    except ValueError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        logger.exception("Analysis failed: %s", exc)
        return 3


def _print_summary(report: dict) -> None:
    """Pretty-print a human-readable analysis summary."""
    eid = report.get("evidence_id", "N/A")
    file_info = report.get("file", {})
    overall = report.get("overall_status", "unknown").upper()

    print(f"  Evidence ID : {eid}")
    print(f"  File Type   : {file_info.get('file_type', 'N/A')}")
    print(f"  SHA-256     : {file_info.get('sha256', 'N/A')[:32]}...")
    print(f"  Status      : {overall}")
    print()

    def _section(title: str, data: dict | None):
        if data is None:
            return
        status = data.get("status", "N/A")
        icon = {"success": "[OK]", "failed": "[FAIL]", "skipped": "[SKIP]"}.get(status, "[?]")
        print(f"  {icon:7s} {title:20s}  status={status}")
        result = data.get("result", {})
        for k, v in result.items():
            if isinstance(v, dict):
                continue   # skip nested
            print(f"       {k}: {v}")
        if data.get("error"):
            print(f"       error: {data['error']}")
        if data.get("skip_reason"):
            print(f"       reason: {data['skip_reason']}")

    _section("Blockchain",     report.get("blockchain"))
    _section("Metadata",       report.get("metadata"))
    _section("Image Forgery",  report.get("image_forgery"))
    _section("Deepfake",       report.get("deepfake"))
    _section("Fake News",      report.get("fake_news"))

    print(f"\n  {'='*56}")
    print(f"  OVERALL: {overall}")
    print(f"  {'='*56}")


def cmd_serve(args: argparse.Namespace) -> int:
    """Start the unified Flask API server."""
    configure_logging("DEBUG" if args.verbose else "INFO")

    from api.app import app

    port = args.port or API_PORT
    host = args.host or API_HOST

    print(f"\n{'='*60}")
    print(f"  AI-DRIVEN FORENSIC SYSTEM — API SERVER")
    print(f"{'='*60}")
    print(f"  Host  : {host}")
    print(f"  Port  : {port}")
    print(f"  URL   : http://{host}:{port}")
    print(f"  Health: http://{host}:{port}/health")
    print(f"{'='*60}\n")

    app.run(host=host, port=port, debug=args.verbose)
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="run_system.py",
        description="AI-Driven Digital Evidence Integrity Monitoring System",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    subparsers = parser.add_subparsers(dest="command")

    # ── analyze sub-command ───────────────────────────────────────────────────
    analyze_parser = subparsers.add_parser("analyze", help="Analyse a single evidence file")
    analyze_parser.add_argument(
        "--file", "-f", required=True,
        help="Path to the evidence file (image, video, or PDF)",
    )
    analyze_parser.add_argument(
        "--user", "-u", default="investigator",
        help="Submitting user/investigator name (default: investigator)",
    )
    analyze_parser.add_argument(
        "--json", "-j", action="store_true",
        help="Print full JSON report to stdout",
    )

    # ── serve sub-command ─────────────────────────────────────────────────────
    serve_parser = subparsers.add_parser("serve", help="Start the unified API server")
    serve_parser.add_argument("--host", default=None, help="Bind host (default: 0.0.0.0)")
    serve_parser.add_argument("--port", "-p", type=int, default=None, help="Port (default: 8080)")

    # ── Backward-compat: allow --file at top level ────────────────────────────
    parser.add_argument("--file", "-f", help="Shortcut: analyse a file without sub-command")
    parser.add_argument("--user", "-u", default="investigator")
    parser.add_argument("--json", "-j", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", "-p", type=int, default=None)

    args = parser.parse_args()

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if args.command == "analyze" or (not args.command and args.file):
        sys.exit(cmd_analyze(args))
    elif args.command == "serve" or (not args.command and args.serve):
        sys.exit(cmd_serve(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
