"""
fake_news_module/similarity/searcher.py
=========================================
Query-time FAISS retrieval for claim similarity search.

Algorithm
---------
    1. Encode the query claim → 384-dim L2-normalized vector
    2. Query the FAISS index for top-K nearest neighbours (cosine similarity)
    3. Retrieve the labels (Real/Fake) of all K neighbours
    4. Majority vote the labels
    5. If vote is tied OR max similarity score < SIMILARITY_MIN_SCORE → "Uncertain"

Lazy singleton pattern (same as roberta_classifier.py):
  • Index loaded only once on first call to search().
  • Thread-safe via a threading.Lock.
  • Gracefully returns "Uncertain" on any failure.

Usage
-----
    from fake_news_module.similarity.searcher import search

    result = search("Senate panel to hold hearing on Equifax hack")
    # → {
    #     "label": "Real",
    #     "confidence": 0.94,
    #     "top_matches": [
    #         {"title": "Senate holds Equifax hearing", "label": "Real", "score": 0.96},
    #         ...
    #     ]
    # }
"""

import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from fake_news_module.cache import get_cache_manager

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
LABEL_MAP: Dict[int, str] = {0: "Fake", 1: "Real"}

# ──────────────────────────────────────────────
# Singleton state
# ──────────────────────────────────────────────
_lock   = threading.Lock()
_index  = None          # faiss.IndexFlatIP
_labels = None          # np.ndarray[int8]
_titles = None          # np.ndarray[object]


def _load_index(index_dir: Optional[str] = None) -> bool:
    """
    Lazily load the FAISS index and label/title arrays.

    Args:
        index_dir: Optional override for the index directory.

    Returns:
        True if loaded successfully, False otherwise.
    """
    global _index, _labels, _titles

    with _lock:
        if _index is not None:
            return True

        try:
            import faiss
        except ImportError as exc:
            logger.error(
                "Searcher: faiss-cpu not installed — %s. "
                "Run: pip install faiss-cpu",
                exc,
            )
            return False

        # Resolve index directory
        from fake_news_module.config import SIMILARITY_INDEX_DIR
        resolved_dir = Path(index_dir or SIMILARITY_INDEX_DIR)

        index_path  = resolved_dir / "faiss_index.bin"
        labels_path = resolved_dir / "labels.npy"
        titles_path = resolved_dir / "titles.npy"

        if not index_path.exists():
            logger.warning(
                "Searcher: index not found at '%s'. "
                "Run: python -m fake_news_module.similarity.index_builder",
                index_path,
            )
            return False

        try:
            logger.info("Searcher: loading FAISS index from '%s' ...", index_path)
            _index  = faiss.read_index(str(index_path))
            _labels = np.load(str(labels_path))
            _titles = np.load(str(titles_path), allow_pickle=True)
            logger.info(
                "Searcher: index loaded. %d vectors, %d labels.",
                _index.ntotal, len(_labels),
            )
            return True

        except Exception as exc:  # noqa: BLE001
            logger.error("Searcher: failed to load index — %s", exc)
            _index = _labels = _titles = None
            return False


def search(
    claim: str,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
    index_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Find the top-K most similar claims in the index and majority-vote their labels.

    Args:
        claim:     The input claim text to search for.
        top_k:     Number of nearest neighbours to retrieve (default from config).
        min_score: Minimum cosine similarity to trust a result (default from config).
        index_dir: Optional override for the index directory.

    Returns:
        Dict with keys:
            "label":       "Real" | "Fake" | "Uncertain"
            "confidence":  float — average cosine similarity of the winning-class matches
            "top_matches": List of {"title": str, "label": str, "score": float}
    """
    _UNCERTAIN = {
        "label":       "Uncertain",
        "confidence":  0.0,
        "top_matches": [],
    }

    if not claim or not claim.strip():
        logger.warning("Searcher.search: empty claim.")
        return _UNCERTAIN

    cache = get_cache_manager()
    cached = cache.get_similarity_result(claim)
    if isinstance(cached, dict):
        logger.info("Searcher.search: cache hit")
        return cached

    # Load config defaults
    try:
        from fake_news_module.config import SIMILARITY_TOP_K, SIMILARITY_MIN_SCORE
        if top_k is None:
            top_k = SIMILARITY_TOP_K
        if min_score is None:
            min_score = SIMILARITY_MIN_SCORE
    except ImportError:
        top_k = top_k or 5
        min_score = min_score or 0.60

    # Ensure index is loaded
    if _index is None:
        if not _load_index(index_dir):
            return _UNCERTAIN

    # Encode query
    from fake_news_module.similarity.embedder import embed
    query_vec = embed([claim])
    if query_vec is None:
        return _UNCERTAIN

    try:
        # FAISS search returns (distances, indices) — distances = inner products
        scores, indices = _index.search(query_vec, top_k)
        scores   = scores[0].tolist()    # shape (top_k,)
        indices  = indices[0].tolist()   # shape (top_k,)

        # Build top matches list
        top_matches: List[Dict] = []
        valid_labels: List[int] = []
        valid_scores: List[float] = []

        for idx, score in zip(indices, scores):
            if idx < 0:   # FAISS returns -1 for empty slots
                continue
            lbl   = int(_labels[idx])
            title = str(_titles[idx]) if _titles is not None else ""
            top_matches.append({
                "title": title,
                "label": LABEL_MAP.get(lbl, "Unknown"),
                "score": round(float(score), 4),
            })
            # Only count matches above the minimum score threshold
            if score >= min_score:
                valid_labels.append(lbl)
                valid_scores.append(score)

        logger.info(
            "Searcher.search: top score=%.4f | %d/%d matches above min_score=%.2f",
            scores[0] if scores else 0.0,
            len(valid_labels), top_k, min_score,
        )

        # Not enough confident matches → Uncertain
        if not valid_labels:
            logger.info("Searcher.search: no matches above threshold — Uncertain")
            result = {**_UNCERTAIN, "top_matches": top_matches}
            cache.set_similarity_result(claim, result)
            return result

        # Majority vote
        real_count = sum(1 for l in valid_labels if l == 1)
        fake_count = sum(1 for l in valid_labels if l == 0)

        if real_count == fake_count:
            # Tie → Uncertain
            voted_label = "Uncertain"
            conf = 0.0
        elif real_count > fake_count:
            voted_label = "Real"
            conf = round(
                float(np.mean([s for l, s in zip(valid_labels, valid_scores) if l == 1])),
                4,
            )
        else:
            voted_label = "Fake"
            conf = round(
                float(np.mean([s for l, s in zip(valid_labels, valid_scores) if l == 0])),
                4,
            )

        logger.info(
            "Searcher.search: vote=Real:%d Fake:%d → label='%s' confidence=%.4f",
            real_count, fake_count, voted_label, conf,
        )

        result = {
            "label":       voted_label,
            "confidence":  conf,
            "top_matches": top_matches,
        }
        cache.set_similarity_result(claim, result)
        return result

    except Exception as exc:  # noqa: BLE001
        logger.error("Searcher.search: retrieval failed — %s", exc)
        return _UNCERTAIN
