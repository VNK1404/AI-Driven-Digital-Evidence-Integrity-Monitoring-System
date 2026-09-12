# AI-Driven Digital Evidence Integrity Monitoring System
## Unified Modular Integration Layer

This document details the unified integration architecture for the **AI-Driven Digital Evidence Integrity Monitoring System**. The system unifies five major forensic analysis modules into a modular, locally runnable backend without replacing trained models or modifying existing codebases.

---

## 1. System Architecture

The project integrates 5 forensic components into a unified execution flow using an **Adapter Pattern** and central **Orchestrator Layer**:

```
[ Input File (Image / Video / PDF) ]
                  │
                  ▼
   ┌──────────────────────────────┐
   │ orchestration/evidence_router│ (Determines category & applicable modules)
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │ evidence_id (EV-XXXXXX)      │ (Deterministic unique identifier & SHA-256)
   └──────────────┬───────────────┘
                  │
   ┌──────────────┼──────────────────┬─────────────────┬─────────────────┐
   │              │                  │                 │                 │
   ▼              ▼                  ▼                 ▼                 ▼
┌──────────────┐┌────────────────┐┌─────────────────┐┌───────────────┐┌───────────────┐
│  Blockchain  ││    Metadata    ││  Image Forgery  ││   Deepfake    ││   Fake News   │
│   Adapter    ││    Adapter     ││     Adapter     ││    Adapter    ││    Adapter    │
└──────┬───────┘└───────┬────────┘└────────┬────────┘└───────┬───────┘└───────┬───────┘
       │                │                  │                 │                │
       └────────────────┴─────────┬────────┴─────────────────┴────────────────┘
                                  ▼
                   ┌─────────────────────────────┐
                   │  analysis_orchestrator.py   │ (Fault-isolated aggregator)
                   └──────────────┬──────────────┘
                                  ▼
                   ┌─────────────────────────────┐
                   │ JSON Report & SQLite Record │ (reports/EV-XXXXXX.json)
                   └─────────────────────────────┘
```

---

## 2. Integrated Forensic Modules

| # | Forensic Module | Adapter Path | Model / Weights | Execution Scope |
|---|-----------------|--------------|-----------------|-----------------|
| 1 | **Blockchain / Custody** | `integrations/blockchain/adapter.py` | SQLite `evidence.db` | All supported files (Image, Video, Document) |
| 2 | **Metadata Forensics** | `integrations/metadata/adapter.py` | Rule Engine + Z-Score Profile | Image, Video, Document (Audio excluded) |
| 3 | **Image Forgery** | `integrations/image_forgery/adapter.py` | ResNet50 + EfficientNet-B0 + LogReg | Images only (`.jpg`, `.png`, `.tif`, etc.) |
| 4 | **Deepfake Detection** | `integrations/deepfake/adapter.py` | EfficientNet-B3 + BiLSTM + Attention | Videos only (`.mp4`, `.avi`, `.mov`, etc.) |
| 5 | **Fake News Detection** | `integrations/fake_news/adapter.py` | RoBERTa + FAISS Index + Groq/APIs | Images & Documents (conditional on text/OCR) |

---

## 3. Directory Layout

```
new_final/
├── config/
│   ├── __init__.py
│   └── settings.py               # Central settings & env loader
├── schemas/
│   ├── __init__.py
│   ├── evidence.py               # EvidenceObject & SHA-256 dataclass
│   └── results.py                # ModuleResult & ForensicReport dataclasses
├── integrations/
│   ├── __init__.py
│   ├── blockchain/adapter.py     # Wraps blockchain_chain_of_custudy/
│   ├── metadata/adapter.py       # Wraps metadata_forensics/
│   ├── image_forgery/adapter.py  # Wraps image_forgery_detection/
│   ├── deepfake/adapter.py       # Wraps deepfake_detector/
│   └── fake_news/adapter.py      # Wraps fake_news_detection_final/
├── orchestration/
│   ├── __init__.py
│   ├── evidence_router.py        # File-type detection & module routing
│   ├── result_aggregator.py      # Aggregates ModuleResults into ForensicReport
│   └── analysis_orchestrator.py  # Main pipeline execution entry point
├── api/
│   ├── __init__.py
│   └── app.py                    # Unified Flask REST API (port 8080)
├── tests/                        # Full test suite
├── reports/                      # Per-evidence JSON output reports
├── logs/                         # Unified system logs
├── run_system.py                 # Primary CLI tool
├── run_dataset_analysis.py       # Batch directory processor
├── requirements.txt              # Merged deduplicated requirements
└── .env.example                  # Environment configuration template
```

---

## 4. Key Design Principles

1. **Adapter Pattern**: Each module is imported as library code via Python adapters using controlled `sys.path` injection.
2. **Single Evidence Identifier**: Every uploaded item receives a single `evidence_id` (`EV-XXXXXX`) passed across all modules.
3. **Fault Isolation**: One module failure will **NEVER** crash the pipeline. Failed modules return `status: "failed"` while others complete.
4. **Dynamic File Routing**: Audio files are explicitly rejected. Video files only trigger Video & Metadata & Blockchain analysis.
5. **Local Persistence**: Reports are persisted to `reports/EV-XXXXXX.json` and custody events logged to SQLite `blockchain_chain_of_custudy/database/evidence.db`. (Supabase is deferred to later phases).

---

## 5. API Reference (Port 8080)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` or `/ui` | GET | Forensic Workbench Testing Web UI |
| `/health` | GET | System health & in-memory blockchain status |
| `/analyze` | POST | Form-data file upload analysis (`file`, optional `submitted_by`) |
| `/analyze/path` | POST | Server-side file path analysis (`{"file_path": "..."}`) |
| `/report/<id>` | GET | Retrieve saved JSON report by Evidence ID |
| `/reports` | GET | List all generated report Evidence IDs |
| `/custody/<id>` | GET | Get audit trail for specific evidence |
| `/approve/<id>` | POST | Admin approval to add evidence block to blockchain |
| `/chain` | GET | View full blockchain ledger |
| `/verify/<id>` | POST | Re-verify file integrity against stored SHA-256 |
