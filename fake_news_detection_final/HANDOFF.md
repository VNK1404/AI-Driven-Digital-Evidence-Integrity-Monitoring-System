# Fake News Detection Engine — Backend Package
## Team Handoff Documentation

---

## 📦 What's Included

This package contains the **complete backend verification engine** for the Multi-Modal Fake News Detection Platform. It is ready for integration with additional ML models or front-end services.

```
fake_news_detector/
├── app.py                        # Flask REST API server (entry point)
├── main.py                       # CLI entry point for batch/terminal analysis
├── requirements.txt              # Python dependencies
├── check_api_status.py           # API key diagnostic utility
├── test_local_backend.py         # Local component health check (no API calls)
├── .env.example                  # Environment variables template
├── pytest.ini                    # Test runner configuration
├── templates/
│   └── index.html                # Optional HTML upload UI
├── tests/                        # Automated test suite (12 test files)
└── fake_news_module/
    ├── config.py                 # Central configuration (weights, thresholds, keys)
    ├── core/
    │   ├── pipeline.py           # End-to-end orchestration
    │   ├── decision_engine.py    # Weighted scoring and verdict thresholds
    │   └── normalization.py      # Label-to-score normalization
    ├── ocr/
    │   ├── extractor.py          # Unified OCR dispatcher
    │   ├── image_ocr.py          # Tesseract-based image OCR
    │   └── pdf_reader.py         # PyMuPDF-based PDF text extraction
    ├── processing/
    │   ├── claim_extractor.py    # Extracts core claim from cleaned text
    │   ├── text_cleaner.py       # Noise removal and text normalization
    │   └── validator.py          # Input validation rules
    ├── ml/
    │   ├── roberta_classifier.py # Fine-tuned RoBERTa inference (lazy singleton)
    │   ├── inference.py          # ML inference helpers
    │   ├── train.py              # Model training script (for retraining)
    │   └── saved_model/          # ⚠️  Model weights NOT included (see note below)
    ├── similarity/
    │   ├── searcher.py           # FAISS query-time retrieval
    │   ├── embedder.py           # all-MiniLM-L6-v2 sentence embeddings
    │   ├── index_builder.py      # Build FAISS index from dataset
    │   ├── inference.py          # Similarity inference helper
    │   └── index/                # Pre-built FAISS index (44,898 claims)
    ├── apis/
    │   ├── async_client.py       # Parallel aiohttp API client pool
    │   ├── google_api.py         # Google Fact Check Tools API
    │   ├── news_api.py           # NewsAPI integration
    │   └── groq_api.py           # Groq API (Llama 3.3 70B)
    ├── source_scoring/
    │   ├── source_scorer.py      # Domain credibility evaluator
    │   └── source_scores.json    # Pre-scored domain credibility database
    ├── explainability/
    │   ├── explanation_engine.py # Verdict reason + evidence chain generator
    │   ├── evidence_builder.py   # Evidence link builder
    │   └── confidence_breakdown.py  # Signal weight % breakdown
    ├── analytics/
    │   ├── metrics_store.py      # SQLite-backed analysis logger
    │   ├── analytics_service.py  # Analytics aggregation service
    │   └── dashboard_api.py      # Flask blueprint for /api/analytics routes
    ├── cache/
    │   ├── cache_manager.py      # High-level cache facade
    │   └── redis_cache.py        # Redis + in-memory cache backend
    └── reporting/
        └── report_generator.py   # PDF + JSON report generator
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites
- Python 3.10+
- Tesseract OCR installed ([Download](https://tesseract-ocr.github.io/tessdoc/Installation.html))
- CUDA (optional — for GPU-accelerated RoBERTa inference)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install "numpy<2" sentence-transformers faiss-cpu
```

### 3. Configure API Keys

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Required keys:
| Variable | Service | Get Key At |
|---|---|---|
| `GOOGLE_FACT_CHECK_API_KEY` | Google Fact Check Tools | [console.cloud.google.com](https://console.cloud.google.com) |
| `NEWS_API_KEY` | NewsAPI.org | [newsapi.org/register](https://newsapi.org/register) |
| `GROQ_API_KEY` | Groq (Llama 3.3 70B) | [console.groq.com](https://console.groq.com) |
| `FLASK_SECRET` | Flask session signing | Generate locally |

### 4. Load the RoBERTa Model Weights

> ⚠️ **Model weights are NOT included** in this package (475 MB — too large for git).
> The team lead should either:
> - Request the `model.safetensors` file separately and place it in `fake_news_module/ml/saved_model/`
> - Or run training from scratch using `python -m fake_news_module.ml.train`

### 5. Verify All Components Work

```bash
# Check API keys
python check_api_status.py

# Check local backend components (no API calls)
python test_local_backend.py
```

### 6. Start the Server

```bash
python app.py
```

The API will be available at: `http://localhost:5000`

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server health check |
| `POST` | `/analyze` | Submit text or file for verification |
| `GET/POST` | `/download_report` | Download PDF report for an analysis |
| `GET` | `/api/analytics` | Summary + performance metrics |
| `GET` | `/api/analytics/summary` | Claim counts breakdown |
| `GET` | `/api/analytics/performance` | Pipeline latency metrics |

### Example `/analyze` Request

```bash
# Text claim
curl -X POST http://localhost:5000/analyze \
  -F "text=Scientists discover water on Mars"

# File upload
curl -X POST http://localhost:5000/analyze \
  -F "file=@/path/to/document.pdf"
```

---

## 🧠 Signal Weights (Decision Engine)

| Signal | Module | Weight |
|---|---|---|
| RoBERTa Classifier | `ml/roberta_classifier.py` | 20% |
| FAISS Similarity | `similarity/searcher.py` | 20% |
| Google Fact Check | `apis/google_api.py` | 20% |
| NewsAPI Coverage | `apis/news_api.py` | 10% |
| Groq LLM | `apis/groq_api.py` | 25% |
| Heuristic Detector | `processing/heuristic_detector.py` | 5% |
| Source Credibility | `source_scoring/` | 10% (bonus) |

Weights are configurable in `fake_news_module/config.py`.

---

## 🔌 Integration Guide for New Models

To plug in a new ML model or API signal:

1. **Add a new API module** under `fake_news_module/apis/your_model.py`
   - Implement a function returning one of: `"Real"`, `"Fake"`, `"Uncertain"`, `"Unknown"`

2. **Register a weight** in `fake_news_module/config.py`:
   ```python
   API_WEIGHTS = {
       ...,
       "your_model": 0.XX,   # adjust weights to sum to 1.0
   }
   ```

3. **Integrate into the pipeline** in `fake_news_module/core/pipeline.py`

4. **Update the async client** in `fake_news_module/apis/async_client.py` to call your model concurrently.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📋 Key Design Decisions

- **Graceful fallback**: Every signal handles failures gracefully — if an API is down, the engine re-normalizes remaining signal weights.
- **Caching**: Redis-backed (with in-memory fallback) to avoid redundant API calls.
- **Async parallel execution**: All external APIs are called concurrently via `aiohttp`.
- **PyTorch backend only**: `USE_TF=0` is set explicitly to prevent Keras/TensorFlow conflicts.
