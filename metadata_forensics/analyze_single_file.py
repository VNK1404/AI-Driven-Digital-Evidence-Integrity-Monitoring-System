#!/usr/bin/env python3
"""
=============================================================================
  analyze_single_file.py — Interactive Single-File Forensics Analyzer
=============================================================================
Allows you to input a file path (image, video, audio, or document),
runs the full zero-training metadata forensics pipeline, and outputs
the score, flags, anomaly detection result, and extracted metadata.
"""

import os
import sys
import json
from pipeline.metadata_pipeline import run_metadata_pipeline

def print_separator(char="-", length=60):
    print(char * length)

def main():
    print_separator("=")
    print("  AI-Driven Metadata Forensics — Single File Analyzer")
    print_separator("=")
    print()

    # Get file path
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        try:
            file_path = input("Enter the path to the evidence file: ").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            sys.exit(0)

    # Remove quotes if user dragged-and-dropped the file into terminal
    if file_path.startswith('"') and file_path.endswith('"'):
        file_path = file_path[1:-1]
    elif file_path.startswith("'") and file_path.endswith("'"):
        file_path = file_path[1:-1]

    if not file_path:
        print("Error: No file path provided.")
        sys.exit(1)

    if not os.path.exists(file_path):
        print(f"\nError: File not found at '{file_path}'")
        sys.exit(1)

    print("\n🔍 Analyzing file ...\n")
    
    try:
        # Run the full pipeline
        # (Passes model=None because the system uses zero-training reference profiles by default!)
        result = run_metadata_pipeline(file_path)
    except Exception as e:
        print(f"❌ Critical error analyzing file: {e}")
        sys.exit(1)

    # Print Summary Results
    print_separator()
    print("  FORENSIC ANALYSIS REPORT")
    print_separator()
    
    file_type = result["metadata"].get("file_type", "Unknown (Auto-detected)")
    # Note: extraction layer returns file_type, but let's check via features if present
    
    score = result.get('metadata_score', 0)
    anomaly_status = "⚠️ ANOMALY DETECTED" if result.get('anomaly') else "✅ Normal"
    
    print(f"File Path    : {file_path}")
    print(f"File Size    : {os.path.getsize(file_path)} bytes")
    print(f"File Type    : {file_type.upper()}")
    print_separator()
    print(f"INTEGRITY SCORE: {score} / 100")
    print(f"AI DETECTION   : {anomaly_status}")
    print_separator()

    # Print Flags
    flags = result.get("flags", [])
    timeline_flags = result.get("timeline_flags", [])
    
    all_flags = flags + timeline_flags
    if all_flags:
        print("\n🚩 SUSPICIOUS FLAGS IDENTIFIED:")
        for flag in all_flags:
            print(f"  - {flag.replace('_', ' ').capitalize()}")
    else:
        print("\n✅ No suspicious metadata flags identified.")

    # Print Extracted Features
    print("\n📊 ENGINEERED FEATURES:")
    features = result.get("features", {})
    if features:
        for k, v in features.items():
            print(f"  {k}: {v}")
    
    # Optionally display raw metadata
    print()
    print_separator("-", 60)
    print()
    show_meta = input("Would you like to see the raw extracted metadata JSON? (y/N): ").strip().lower()
    
    if show_meta in ['y', 'yes']:
        print("\n📋 RAW METADATA:")
        # Pretty-print JSON, handling non-serializable objects by converting them to string
        print(json.dumps(result.get("metadata", {}), indent=4, default=str))

if __name__ == "__main__":
    main()
