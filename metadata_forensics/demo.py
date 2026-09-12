import os
import glob
import json
from pipeline.metadata_pipeline import run_metadata_pipeline

# Find one of each file type
base_dir = "datasets"
files_to_test = {
    "image": glob.glob(f"{base_dir}/**/*.jpg", recursive=True),
    "video": glob.glob(f"{base_dir}/**/*.mp4", recursive=True),
    "audio": glob.glob(f"{base_dir}/**/*.wav", recursive=True),
    "document": glob.glob(f"{base_dir}/**/*.pdf", recursive=True)
}

print("==================================================")
print("Metadata Forensics Pipeline - Demonstration")
print("==================================================\n")

all_results = {}

for file_type, file_list in files_to_test.items():
    if not file_list:
        print(f"Skipping {file_type}: No files found.")
        continue
    
    test_file = file_list[0]
    print(f"Testing {file_type.upper()} File:")
    print(f"Path: {test_file}")
    
    # Run the pipeline
    result = run_metadata_pipeline(test_file)
    
    # Print the summary
    print(f"Score:   {result['metadata_score']}/100")
    print(f"Anomaly: {result['anomaly']}")
    print(f"Flags:   {result['flags']}")
    print(f"Extracted {len(result['metadata'])} metadata fields.")
    print("-" * 50)
    
    all_results[file_type] = result

# Save all results to a JSON file so the user can inspect the raw metadata
output_file = "demo_extraction_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=4, default=str)

print(f"\n✅ All extracted metadata has been saved to '{output_file}' for you to inspect.")
