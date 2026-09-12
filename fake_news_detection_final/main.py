"""
main.py — Entry point for Fake News Detection Module
AI-Driven Digital Evidence Integrity Monitoring System

Run in VS Code terminal:
    python main.py

A native file-picker dialog will open.
You can also pass a file directly:
    python main.py --file path/to/file.jpg
"""

import sys
import json
import argparse
import logging
from pathlib import Path

# ── Ensure project root is importable ──────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_news_module.core.pipeline import fake_news_pipeline
from fake_news_module.config import LOG_FORMAT, LOG_DATE_FORMAT, LOG_LEVEL

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Native file-picker (tkinter)
# Works in VS Code integrated terminal on Windows
# ──────────────────────────────────────────────────────────────
def pick_file_with_dialog() -> str:
    """
    Open a native Windows file-picker dialog and return the chosen path.
    Falls back to a manual console input if tkinter is unavailable.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        # Hide the root Tk window — we only want the dialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)   # Bring dialog to front

        print("\n[INFO] A file picker dialog is opening — select your image or PDF ...\n")

        file_path = filedialog.askopenfilename(
            title="Select Image or PDF for Fake News Detection",
            filetypes=[
                ("Supported files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp *.pdf"),
                ("Images",          "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("PDF Documents",   "*.pdf"),
                ("All files",       "*.*"),
            ],
        )
        root.destroy()

        if not file_path:
            print("[WARN] No file selected. Exiting.")
            sys.exit(0)

        return file_path

    except ImportError:
        # tkinter not available — ask via terminal
        print("[WARN] tkinter not available. Falling back to manual input.")
        file_path = input("Enter the full path to your image or PDF file:\n> ").strip().strip('"')
        if not file_path:
            print("[ERROR] No path provided. Exiting.")
            sys.exit(0)
        return file_path


# ──────────────────────────────────────────────────────────────
# Result formatting (terminal output)
# ──────────────────────────────────────────────────────────────
def print_result(result: dict) -> None:
    decision   = result.get("final_decision", "Uncertain")
    confidence = result.get("confidence", "Low")
    score      = result.get("score", 0.0)
    claim      = result.get("claim", "")
    apis       = result.get("api_results", {})

    ICONS = {"Real": "[REAL]", "Fake": "[FAKE]", "Uncertain": "[UNCERTAIN]"}
    icon  = ICONS.get(decision, "[?]")

    sep = "-" * 56

    print()
    print(sep)
    print(f"  VERDICT    : {icon} {decision.upper()}")
    print(f"  CONFIDENCE : {confidence}")
    print(f"  SCORE      : {score:+.4f}  (range: -1.0 Fake ... +1.0 Real)")
    print(sep)
    print(f"  CLAIM      : {claim[:120]}")
    print(sep)
    print("  API VERDICTS:")
    print(f"    RoBERTa (Local)    : {apis.get('roberta', 'Unknown')}")
    print(f"    FAISS Similarity   : {apis.get('similarity', 'Uncertain')}")
    print(f"    Google Fact Check  : {apis.get('google_fact_check', 'Unknown')}")
    print(f"    NewsAPI            : {apis.get('news_api', 'Unknown')}")
    print(f"    Groq LLM (Llama 3.3): {apis.get('groq', 'Unknown')}")
    print(f"    Source Credibility : {apis.get('source', 'Uncertain')}")
    print(sep)
    print()


# ──────────────────────────────────────────────────────────────
# CLI argument parser (optional — dialog used when --file omitted)
# ──────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fake_news_detector",
        description=(
            "Multi-Modal Fake News Detection Engine.\n"
            "Run without arguments to open a file-picker dialog.\n"
            "Or pass --file directly."
        ),
    )
    parser.add_argument(
        "--file", "-f",
        default=None,
        metavar="FILE_PATH",
        help="Path to image (.jpg/.png) or PDF. Omit to open file-picker.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="OUTPUT_JSON",
        help="Save result as JSON to this path (optional).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="Only print final JSON, suppress logs.",
    )
    return parser


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main() -> None:
    parser  = build_parser()
    args    = parser.parse_args()

    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    print()
    print("  Fake News Detection Engine")
    print("  AI-Driven Digital Evidence Integrity Monitoring System")
    print()

    # ── Get file path: dialog OR --file arg ────────────────────
    if args.file:
        file_path = str(Path(args.file).resolve())
        print(f"[INFO] File: {file_path}")
    else:
        file_path = pick_file_with_dialog()
        file_path = str(Path(file_path).resolve())
        print(f"[INFO] Selected: {file_path}")

    print("[INFO] Running analysis pipeline ...\n")

    # ── Run pipeline ───────────────────────────────────────────
    try:
        result = fake_news_pipeline(file_path)
    except FileNotFoundError as exc:
        print(f"[ERROR] File not found: {exc}")
        sys.exit(1)
    except ValueError as exc:
        print(f"[ERROR] Validation failed: {exc}")
        sys.exit(2)
    except RuntimeError as exc:
        print(f"[ERROR] Runtime error: {exc}")
        sys.exit(3)
    except Exception as exc:                          # noqa: BLE001
        logger.exception("Unexpected error: %s", exc)
        sys.exit(99)

    # ── Print result ──────────────────────────────────────────
    print_result(result)

    # ── Optional JSON save ────────────────────────────────────
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[INFO] Result saved to: {out.resolve()}")
    else:
        # Always print JSON at the end
        print("---- JSON OUTPUT ----")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    # ── Keep terminal open so user can read results ───────────
    print()
    input("Press Enter to exit ...")


if __name__ == "__main__":
    main()
