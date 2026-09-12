# Metadata Forensics — Production-Ready Module

## Overview

A **zero-training** metadata forensic analysis pipeline for digital evidence.
Pass in **any** file (image, video, audio, PDF) and get an integrity score,
rule-based flags, and AI anomaly detection — **no dataset training required**.

---

## 1. Prerequisites

- **Python 3.8+** (recommended: 3.10+)
- **pip** package manager
- **ExifTool** (optional — improves video metadata extraction)
  - Download from https://exiftool.org/ and add to PATH
- **FFmpeg/ffprobe** (optional — fallback for video when ExifTool is absent)

---

## 2. Install Dependencies

```bash
cd metadata_forensics
pip install -r requirements.txt
```

---

## 3. Quick Start — Analyse a Single File

```python
from pipeline.metadata_pipeline import run_metadata_pipeline

result = run_metadata_pipeline("path/to/evidence.jpg")

print(result["metadata_score"])  # 0–100 integrity score
print(result["flags"])           # ['editing_software_detected', ...]
print(result["anomaly"])         # True / False
```

---

## 4. Module API Reference

### 4.1 Unified Extraction

```python
from extraction import extract_metadata

result = extract_metadata("photo.jpg")
# {"file_type": "image", "metadata": {...}}
```

Supported types: `.jpg`, `.png`, `.tif`, `.bmp`, `.mp4`, `.avi`, `.mov`, `.mkv`,
`.wav`, `.mp3`, `.flac`, `.ogg`, `.pdf`

### 4.2 Feature Engineering

```python
from analysis.features import metadata_to_features

features = metadata_to_features(metadata, file_type="image")
# {"camera_present": 1, "editing_software_present": 0, "gps_present": 1,
#  "metadata_length": 22, "device_info_present": 1, "timestamp_present": 1,
#  "has_error": 0}
```

### 4.3 Rule-Based Analysis

```python
from analysis.rules import metadata_rules

flags = metadata_rules(metadata, file_type="image")
# ["camera_model_missing", "editing_software_detected"]
```

### 4.4 AI Anomaly Detection

```python
from analysis.anomaly_model import detect_anomaly, train_anomaly_model

# Zero-training (works immediately, no data needed)
is_anomaly = detect_anomaly(None, feature_vector, file_type="image")

# Optional: train on batch data for higher accuracy
model = train_anomaly_model(list_of_feature_vectors)
is_anomaly = detect_anomaly(model, feature_vector)
```

### 4.5 Scoring

```python
from analysis.scoring import compute_metadata_score

score = compute_metadata_score(flags, anomaly=True)
# 100 - (20 × flags) - (30 if anomaly) → clamped to [0, 100]
```

### 4.6 Full Pipeline

```python
from pipeline.metadata_pipeline import run_metadata_pipeline

result = run_metadata_pipeline("evidence.pdf", model=None)
# {
#   "metadata_score": 60,
#   "flags": ["author_missing", "creator_missing"],
#   "anomaly": False,
#   "metadata": {...},
#   "features": {...},
#   "timeline_flags": [...]
# }
```

---

## 5. Batch Processing

To process entire datasets:

```bash
python run_full_metadata_forensics.py
```

Results are saved to `forensics_results_full.csv`. If interrupted, re-run to resume.

---

## 6. Run Tests

```bash
python test_pipeline.py
```

---

## 7. How Anomaly Detection Works (No Training Required)

The system uses **statistical reference profiles** — pre-computed distributions of
"normal" metadata for each file type (image, video, audio, document). Each file's
metadata features are compared against these profiles using z-score deviation.

If the average deviation exceeds the threshold (z > 1.8), the file is flagged as
anomalous.

For **higher accuracy**, you can optionally train an IsolationForest model on your
own dataset using `train_anomaly_model()`.

---

## 8. Troubleshooting

| Issue | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Video metadata empty | Install ExifTool or FFmpeg |
| Audio errors on MP3 | Ensure `mutagen` is installed |
| Score always 0 | Normal for heavily edited or metadata-stripped files |
