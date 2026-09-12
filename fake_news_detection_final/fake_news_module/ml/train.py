"""
fake_news_module/ml/train.py
============================
Fine-tuning pipeline for the RoBERTa-based fake-news binary classifier.

Dataset format (CSV)
--------------------
    text,label
    "Scientists discover water on Mars",1
    "Government secretly poisoning water supply",0
    ...
    label:  0 = Fake
            1 = Real

Usage
-----
    # From the project root directory:
    python -m fake_news_module.ml.train \\
        --dataset path/to/data.csv \\
        --output  fake_news_module/ml/saved_model \\
        --epochs  3 \\
        --batch   16

Recommended public datasets
----------------------------
    • LIAR      (Kaggle / HuggingFace datasets hub)
    • FakeNewsNet
    • WELFake   — 72k labelled articles, ready-to-use CSV

Outputs
-------
    saved_model/          — HuggingFace model + tokenizer artefacts
    training_metrics.json — per-epoch accuracy, precision, recall, F1
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ──────────────────────────────────────────────
# Metric computation (used by HuggingFace Trainer)
# ──────────────────────────────────────────────

def _compute_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute accuracy, precision, recall and F1 from Trainer eval output.

    Called automatically by `Trainer.evaluate()` after each epoch.
    """
    try:
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )
    except ImportError as exc:
        raise ImportError("Install scikit-learn: pip install scikit-learn") from exc

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc  = accuracy_score(labels, predictions)
    prec = precision_score(labels, predictions, average="binary", zero_division=0)
    rec  = recall_score(labels, predictions, average="binary", zero_division=0)
    f1   = f1_score(labels, predictions, average="binary", zero_division=0)

    logger.info(
        "Eval — Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f",
        acc, prec, rec, f1,
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ──────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────

def _load_dataset(csv_path: str, tokenizer, max_length: int = 512):
    """
    Load a CSV file (columns: `text`, `label`) and return a HuggingFace Dataset.

    Labels are expected to be integers:  0 = Fake, 1 = Real.

    Args:
        csv_path:   Path to the CSV file.
        tokenizer:  HuggingFace tokenizer instance.
        max_length: Maximum token sequence length.

    Returns:
        HuggingFace DatasetDict with 'train' and 'test' splits (80/20).
    """
    try:
        from datasets import Dataset
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Install datasets and pandas: pip install datasets pandas"
        ) from exc

    logger.info("Loading dataset from: %s", csv_path)
    df = pd.read_csv(csv_path)

    # Validate required columns
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"CSV must contain 'text' and 'label' columns. Found: {list(df.columns)}"
        )

    # Drop rows with missing values
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    logger.info(
        "Dataset: %d rows | Fake(0)=%d | Real(1)=%d",
        len(df),
        (df["label"] == 0).sum(),
        (df["label"] == 1).sum(),
    )

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))

    # Tokenise
    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    dataset = dataset.map(_tokenize, batched=True)
    dataset = dataset.rename_column("label", "labels")
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # 80/20 train/test split
    split = dataset.train_test_split(test_size=0.2, seed=42)
    logger.info(
        "Split: train=%d | test=%d", len(split["train"]), len(split["test"])
    )
    return split


# ──────────────────────────────────────────────
# Training entry-point
# ──────────────────────────────────────────────

def train(
    dataset_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 512,
) -> Dict[str, float]:
    """
    Fine-tune `roberta-base` on the provided CSV dataset.

    Args:
        dataset_path:  Path to training CSV (columns: text, label).
        output_dir:    Directory to save the fine-tuned model and tokenizer.
        epochs:        Number of training epochs (default 3).
        batch_size:    Per-device training batch size (default 16).
        learning_rate: AdamW learning rate (default 2e-5).
        max_length:    Max token length (default 512).

    Returns:
        Dict with final evaluation metrics.
    """
    try:
        from transformers import (
            RobertaTokenizer,
            RobertaForSequenceClassification,
            TrainingArguments,
            Trainer,
        )
    except ImportError as exc:
        raise ImportError("Install transformers: pip install transformers") from exc

    # ── Load tokenizer & backbone ─────────────────────────────────
    logger.info("Loading roberta-base tokenizer …")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    logger.info("Loading roberta-base model with 2 classification labels …")
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base",
        num_labels=2,
        id2label={0: "Fake", 1: "Real"},
        label2id={"Fake": 0, "Real": 1},
    )

    # ── Load & tokenise dataset ───────────────────────────────────
    split = _load_dataset(dataset_path, tokenizer, max_length)

    # ── Training arguments ────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",  # Disable wandb/tensorboard unless configured
    )

    # ── Trainer ───────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        compute_metrics=_compute_metrics,
    )

    logger.info("Starting training …")
    trainer.train()

    # ── Save model + tokenizer ────────────────────────────────────
    logger.info("Saving model to '%s' …", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ── Final evaluation ──────────────────────────────────────────
    logger.info("Running final evaluation …")
    metrics = trainer.evaluate()

    # Persist metrics to JSON
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Metrics saved to '%s'.", metrics_path)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("  Accuracy  : %.4f", metrics.get("eval_accuracy", 0))
    logger.info("  Precision : %.4f", metrics.get("eval_precision", 0))
    logger.info("  Recall    : %.4f", metrics.get("eval_recall", 0))
    logger.info("  F1        : %.4f", metrics.get("eval_f1", 0))
    logger.info("=" * 60)

    return metrics


# ──────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune RoBERTa for fake-news binary classification."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to CSV dataset (columns: text, label  where label 0=Fake 1=Real)",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent / "saved_model"),
        help="Directory to save fine-tuned model (default: fake_news_module/ml/saved_model/)",
    )
    parser.add_argument("--epochs",    type=int,   default=3,    help="Training epochs (default: 3)")
    parser.add_argument("--batch",     type=int,   default=16,   help="Batch size per device (default: 16)")
    parser.add_argument("--lr",        type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--max_len",   type=int,   default=512,  help="Max token length (default: 512)")

    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        max_length=args.max_len,
    )
