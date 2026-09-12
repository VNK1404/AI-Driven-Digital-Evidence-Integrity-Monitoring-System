$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

Write-Host ""
Write-Host "Testing REAL sample..."
python .\predict.py --video ".\test_videos\real_sample.mp4"

Write-Host ""
Write-Host "Testing FAKE sample..."
python .\predict.py --video ".\test_videos\fake_sample.mp4"
