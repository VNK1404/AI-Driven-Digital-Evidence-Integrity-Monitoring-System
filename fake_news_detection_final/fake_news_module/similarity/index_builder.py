"""
fake_news_module/similarity/index_builder.py
==============================================
Builds and persists a FAISS inner-product index over all article titles
in the training corpus (dataset/combined.csv — 44,898 rows).

Index design
------------
    IndexFlatIP:  exact brute-force inner-product (= cosine similarity on
                  L2-normalized vectors). Chosen over IVF because:
                    • dataset fits comfortably in RAM (~17 MB at 384-dim float32)
                    • no training step required
                    • deterministic, exact results every time

Outputs (saved to SIMILARITY_INDEX_DIR)
-------
    faiss_index.bin  — serialized FAISS index
    labels.npy       — int8 array mapping row index → label (0=Fake, 1=Real)
    titles.npy       — object array of title strings (for human-readable top matches)

Usage
-----
    # From the project root:
    python -m fake_news_module.similarity.index_builder

    # Custom dataset path:
    python -m fake_news_module.similarity.index_builder --dataset path/to/combined.csv
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _load_corpus(csv_path: str) -> Tuple[list, np.ndarray]:
    """
    Load titles and labels from the combined CSV dataset.

    Args:
        csv_path: Path to dataset CSV with columns 'text' and 'label'.

    Returns:
        Tuple of (titles: List[str], labels: np.ndarray[int8])
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required: pip install pandas") from exc

    logger.info("Loading corpus from: %s", csv_path)
    df = pd.read_csv(csv_path)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"CSV must contain 'text' and 'label' columns. Found: {list(df.columns)}"
        )

    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    titles = df["text"].tolist()
    labels = df["label"].values.astype(np.int8)

    logger.info(
        "Corpus loaded: %d rows | Fake(0)=%d | Real(1)=%d",
        len(titles),
        int((labels == 0).sum()),
        int((labels == 1).sum()),
    )
    return titles, labels


def build_index(dataset_path: str, output_dir: str, batch_size: int = 512) -> None:
    """
    Build the FAISS index and persist all artefacts.

    Args:
        dataset_path: Path to the combined CSV (text + label columns).
        output_dir:   Directory to save faiss_index.bin, labels.npy, titles.npy.
        batch_size:   Embedding batch size (tune for RAM).
    """
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "faiss-cpu is required: pip install faiss-cpu"
        ) from exc

    from fake_news_module.similarity.embedder import embed, EMBEDDING_DIM

    os.makedirs(output_dir, exist_ok=True)

    # ── Load corpus ──────────────────────────────────────────────
    titles, labels = _load_corpus(dataset_path)

    # ── Encode all titles ────────────────────────────────────────
    logger.info("Encoding %d titles in batches of %d ...", len(titles), batch_size)
    t0 = time.perf_counter()

    all_vectors = []
    for start in range(0, len(titles), batch_size):
        batch = titles[start : start + batch_size]
        vecs = embed(batch, batch_size=batch_size)
        if vecs is None:
            raise RuntimeError(
                "Embedding failed during index build. "
                "Check that sentence-transformers is installed."
            )
        all_vectors.append(vecs)
        done = min(start + batch_size, len(titles))
        logger.info("  Encoded %d / %d titles ...", done, len(titles))

    vectors = np.vstack(all_vectors).astype(np.float32)
    elapsed = time.perf_counter() - t0
    logger.info(
        "Encoding complete in %.1fs. Vectors shape: %s", elapsed, vectors.shape
    )

    # ── Build FAISS index ────────────────────────────────────────
    logger.info("Building FAISS IndexFlatIP (dim=%d) ...", EMBEDDING_DIM)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)
    logger.info("Index built. Total vectors: %d", index.ntotal)

    # ── Save artefacts ───────────────────────────────────────────
    index_path  = os.path.join(output_dir, "faiss_index.bin")
    labels_path = os.path.join(output_dir, "labels.npy")
    titles_path = os.path.join(output_dir, "titles.npy")

    faiss.write_index(index, index_path)
    np.save(labels_path, labels)
    np.save(titles_path, np.array(titles, dtype=object))

    logger.info("Saved FAISS index  → %s", index_path)
    logger.info("Saved labels array → %s", labels_path)
    logger.info("Saved titles array → %s", titles_path)
    logger.info("=" * 60)
    logger.info("INDEX BUILD COMPLETE")
    logger.info("  Vectors : %d", index.ntotal)
    logger.info("  Dim     : %d", EMBEDDING_DIM)
    logger.info("  Output  : %s", output_dir)
    logger.info("=" * 60)


# ──────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build FAISS similarity index from the training dataset."
    )
    parser.add_argument(
        "--dataset",
        default="dataset/combined.csv",
        help="Path to combined CSV (columns: text, label). Default: dataset/combined.csv",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "index"),
        help="Directory to save index artefacts. Default: fake_news_module/similarity/index/",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=512,
        help="Embedding batch size (default: 512).",
    )
    args = parser.parse_args()

    build_index(
        dataset_path=args.dataset,
        output_dir=args.output,
        batch_size=args.batch,
    )
