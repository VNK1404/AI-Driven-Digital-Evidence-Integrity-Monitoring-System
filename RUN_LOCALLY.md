# Running the System Locally

This guide provides step-by-step instructions to set up, run, test, and use the **AI-Driven Digital Evidence Integrity Monitoring System** locally.

---

## 1. Prerequisites

1. **Python**: Python 3.10+ (64-bit).
2. **PyTorch**: PyTorch 2.0+ (CUDA optional, CPU supported).
3. **Tesseract OCR** (System-level dependency required for Fake News OCR):
   - **Windows**: Install via `winget install UB-Mannheim.TesseractOCR` or download installer.
   - Update `TESSERACT_CMD` in `.env` if installed outside default location (`C:\Program Files\Tesseract-OCR\tesseract.exe`).

---

## 2. Environment Setup

```bash
# Navigate to project directory
cd "D:\AI-Driven Digital Evidence Integrity Monitoring System for Secure Forensic Investigation\new_final"

# Create a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
copy .env.example .env
```

---

## 3. Command-Line Usage (`run_system.py`)

### Analyse a Single File
```bash
# Image Analysis
python run_system.py --file metadata_forensics/dummy.jpg --user investigator_1

# Video Analysis
python run_system.py --file deepfake_detector/test_videos/real_sample.mp4

# Output Raw JSON to stdout
python run_system.py --file metadata_forensics/dummy.jpg --json
```

### Start Unified REST API Server & Web UI Workbench
```bash
python run_system.py --serve
# Serves API & Web UI on http://localhost:8080
```
Open `http://localhost:8080/` in any browser to access the **Forensic Workbench Web Testing UI**.

---

## 4. Batch Analysis (`run_dataset_analysis.py`)

Run forensic analysis across an entire folder of evidence files:

```bash
# Process a folder of evidence files
python run_dataset_analysis.py --dir "path/to/evidence_folder"

# Filter by file category and export CSV summary
python run_dataset_analysis.py --dir "path/to/folder" --type image --csv summary.csv
```

---

## 5. API Testing (cURL / PowerShell)

### Health Check
```powershell
Invoke-RestMethod -Uri "http://localhost:8080/health" -Method GET
```

### Upload & Analyze File
```powershell
$form = @{
    file = Get-Item "metadata_forensics/dummy.jpg"
    submitted_by = "officer_bob"
}
Invoke-RestMethod -Uri "http://localhost:8080/analyze" -Method POST -Form $form
```

### Retrieve Report
```powershell
Invoke-RestMethod -Uri "http://localhost:8080/report/EV-XXXXXX" -Method GET
```

---

## 6. Running Tests

```bash
# Run unit tests (Fast, no model weights required)
pytest tests/test_router.py tests/test_evidence.py -v

# Run adapter & integration tests
pytest tests/test_blockchain.py tests/test_metadata.py -v

# Run full end-to-end suite (Loads real ML models)
pytest tests/test_end_to_end.py -v
```

---

## 7. Troubleshooting

- **`numpy` version mismatch**: Ensure `numpy<2.0.0` is installed (`pip install "numpy<2.0.0"`).
- **Audio file error**: Audio is explicitly unsupported. System will reject `.mp3`, `.wav`, etc., with clear status messages.
- **Model load latency**: Initial model load for Deepfake or Image Forgery on CPU takes ~10–15 seconds. Models remain cached in memory for subsequent requests.
