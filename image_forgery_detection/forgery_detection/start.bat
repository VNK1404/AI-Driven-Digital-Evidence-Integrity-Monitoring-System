@echo off
:: ──────────────────────────────────────────────────────
::  Image Forgery Detection — Standalone API Server
::  Double-click this file to start the server.
:: ──────────────────────────────────────────────────────

title Image Forgery Detection API

:: Fix OpenMP duplicate library conflict (Anaconda + PyTorch on Windows)
set KMP_DUPLICATE_LIB_OK=TRUE

:: Move to project root (parent of this folder)
cd /d "%~dp0.."

echo.
echo  ============================================================
echo   Image Forgery Detection API - Starting...
echo   URL  : http://localhost:8000
echo   Docs : http://localhost:8000/docs
echo  ============================================================
echo.

python forgery_detection/run_server.py

echo.
echo  Server stopped. Press any key to exit.
pause >nul
