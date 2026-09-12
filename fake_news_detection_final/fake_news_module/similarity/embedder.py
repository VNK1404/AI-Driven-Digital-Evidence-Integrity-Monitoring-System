"""
fake_news_module/similarity/embedder.py
=========================================
Sentence embedding wrapper using sentence-transformers.

Architecture
------------
    Input text(s)
        └─► SentenceTransformer('all-MiniLM-L6-v2')
                └─► 384-dim dense embedding
                        └─► L2 normalization  ← enables cosine similarity via dot-product

The module implements a **lazy singleton** pattern identical to roberta_classifier.py:
  • Model is loaded only once on first call to embed().
  • Thread-safe via a threading.Lock.
  • Gracefully returns None on import or load failure so the pipeline degrades cleanly.

Model choice: all-MiniLM-L6-v2
  • 80 MB — 6x smaller than roberta-base
  • 384-dim embeddings — fast FAISS search
  • State-of-the-art for short-text semantic similarity
  • No GPU required — runs efficiently on CPU

Usage
-----
    from fake_news_module.similarity.embedder import embed

    vecs = embed(["Senate panel to hold hearing on Equifax hack"])
    # → np.ndarray shape (1, 384), dtype float32, L2-normalized
"""

import logging
import threading
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Singleton state
# ──────────────────────────────────────────────
_lock = threading.Lock()
_model = None   # SentenceTransformer instance

EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384


def _load_model() -> bool:
    """
    Lazily import and initialize the SentenceTransformer model.

    Returns:
        True if model loaded successfully, False otherwise.
    """
    global _model

    with _lock:
        if _model is not None:
            return True

        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Embedder: loading model '%s' ...", EMBEDDING_MODEL_NAME)
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            logger.info(
                "Embedder: model loaded. Embedding dim = %d.", EMBEDDING_DIM
            )
            return True

        except ImportError as exc:
            logger.error(
                "Embedder: sentence-transformers not installed — %s. "
                "Run: pip install sentence-transformers",
                exc,
            )
            return False

        except Exception as exc:  # noqa: BLE001
            logger.error("Embedder: failed to load model — %s", exc)
            return False


def embed(texts: List[str], batch_size: int = 256) -> Optional[np.ndarray]:
    """
    Encode a list of text strings into L2-normalized float32 embeddings.

    Args:
        texts:      List of strings to encode.
        batch_size: Number of texts to encode in each forward pass.

    Returns:
        np.ndarray of shape (len(texts), 384), dtype float32, L2-normalized.
        Returns None if the model is unavailable or encoding fails.
    """
    if not texts:
        logger.warning("Embedder.embed: empty input list.")
        return None

    if _model is None:
        if not _load_model():
            return None

    try:
        # encode() returns a numpy array by default; normalize_embeddings=True
        # applies L2 normalization so inner-product == cosine similarity.
        vectors = _model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    except Exception as exc:  # noqa: BLE001
        logger.error("Embedder.embed: encoding failed — %s", exc)
        return None
