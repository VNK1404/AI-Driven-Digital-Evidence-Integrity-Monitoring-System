# Training Progress Report
## EfficientNet-B3 + BiLSTM + Attention — Deepfake Detection

---

## Overall Summary

| Property | Value |
|---|---|
| **Architecture** | EfficientNet-B3 + BiLSTM + Temporal Attention |
| **Dataset** | FaceForensics++ (FF++) |
| **Total Videos** | 400 (200 Real, 200 Fake) |
| **Train / Val Split** | 320 / 80 videos |
| **Total Epochs Run** | **86** |
| **Epochs with Metrics Recorded** | **52** |
| **Best Val Accuracy (ever)** | **86.25%** (Epoch ~50) |
| **Training Phases** | Phase 1 (Frozen backbone): Epochs 1–8 · Phase 2 (Fine-tuning): Epochs 9–86 |
| **Checkpoint Saved At** | Epoch 86 |

> [!IMPORTANT]
> The checkpoint at `models/checkpoint_ffpp.pth` is saved at epoch **86**, but the full per-epoch metrics history only covers **52 epochs**. This means the last ~34 epochs ran but their per-epoch history was not fully captured in the checkpoint (only the per-batch losses were streamed to the log file). The overall best validation accuracy of **86.25%** is what the best model (`best_model_ffpp.pth`) was saved at.

---

## Per-Epoch Metrics (All 52 Recorded Epochs)

| Epoch | Train Loss | Val Acc (%) | Precision (%) | Recall (%) | F1 (%) | AUC-ROC (%) |
|---|---|---|---|---|---|---|
| 1 | 1.1090 | 55.00 | 61.54 | 53.33 | 57.14 | 56.38 |
| 2 | 0.8374 | 50.00 | 57.58 | 42.22 | 48.72 | 56.19 |
| 3 | 0.8583 | 52.50 | 62.96 | 37.78 | 47.22 | 62.10 |
| 4 | 0.8519 | 61.25 | 67.50 | 60.00 | 63.53 | 67.71 |
| 5 | 0.8120 | 51.25 | 75.00 | 20.00 | 31.58 | 70.70 |
| 6 | 0.8061 | 56.25 | 77.78 | 31.11 | 44.44 | 70.54 |
| 7 | 0.8041 | 58.75 | 71.43 | 44.44 | 54.79 | 72.00 |
| 8 | 0.7176 | 58.75 | 80.00 | 35.56 | 49.23 | 78.13 |
| **9** *(Phase 2 start)* | 0.6378 | 63.75 | 80.77 | 46.67 | 59.15 | 78.76 |
| 10 | 0.6267 | 61.25 | 81.82 | 40.00 | 53.73 | 76.83 |
| 11 | 0.5759 | 62.50 | 80.00 | 44.44 | 57.14 | 76.38 |
| 12 | 0.6269 | 63.75 | 83.33 | 44.44 | 57.97 | 78.76 |
| 13 | 0.6067 | 67.50 | 77.14 | 60.00 | 67.50 | 75.90 |
| 14 | 0.6053 | 68.75 | 83.33 | 55.56 | 66.67 | 77.78 |
| 15 | 0.5776 | 62.50 | 80.00 | 44.44 | 57.14 | 79.30 |
| 16 | 0.6333 | 67.50 | 85.19 | 51.11 | 63.89 | 79.17 |
| 17 | 0.6145 | 73.75 | 87.50 | 62.22 | 72.73 | 80.95 |
| 18 | 0.6148 | 72.50 | 82.86 | 64.44 | 72.50 | 79.17 |
| 19 | 0.5618 | 73.75 | 81.58 | 68.89 | 74.70 | 80.44 |
| 20 | 0.5618 | 68.75 | 85.71 | 53.33 | 65.75 | 83.24 |
| 21 | 0.5956 | 73.75 | 85.29 | 64.44 | 73.42 | 80.95 |
| 22 | 0.5484 | 72.50 | 81.08 | 66.67 | 73.17 | 82.22 |
| 23 | 0.5559 | 68.75 | 83.33 | 55.56 | 66.67 | 82.35 |
| 24 | 0.5237 | 76.25 | 82.50 | 73.33 | 77.65 | 87.17 |
| 25 | 0.4822 | 73.75 | 85.29 | 64.44 | 73.42 | 86.41 |
| 26 | 0.5545 | 73.75 | 87.50 | 62.22 | 72.73 | 86.10 |
| 27 | 0.5080 | 76.25 | 86.11 | 68.89 | 76.54 | 87.08 |
| 28 | 0.4972 | 76.25 | 86.11 | 68.89 | 76.54 | 87.52 |
| 29 | 0.4864 | 80.00 | 85.37 | 77.78 | 81.40 | 87.75 |
| 30 | 0.5497 | 78.75 | 88.89 | 71.11 | 79.01 | 88.16 |
| 31 | 0.4731 | 82.50 | 87.80 | 80.00 | 83.72 | 88.44 |
| 32 | 0.4955 | 78.75 | 88.89 | 71.11 | 79.01 | 88.70 |
| 33 | 0.5077 | 80.00 | 93.94 | 68.89 | 79.49 | 90.22 |
| 34 | 0.4778 | 77.50 | 93.55 | 64.44 | 76.32 | 90.13 |
| 35 | 0.4973 | 82.50 | 91.89 | 75.56 | 82.93 | 91.21 |
| 36 | 0.5021 | 80.00 | 89.19 | 73.33 | 80.49 | 90.22 |
| 37 | 0.4723 | 78.75 | 93.75 | 66.67 | 77.92 | 89.84 |
| 38 | 0.4811 | 78.75 | 93.75 | 66.67 | 77.92 | 90.22 |
| 39 | 0.4546 | 78.75 | 93.75 | 66.67 | 77.92 | 91.30 |
| 40 | 0.4706 | 83.75 | 90.00 | 80.00 | 84.71 | 90.16 |
| 41 | 0.4646 | 82.50 | 91.89 | 75.56 | 82.93 | 90.48 |
| 42 | 0.4941 | 80.00 | 91.43 | 71.11 | 80.00 | 90.22 |
| 43 | 0.4794 | 80.00 | 91.43 | 71.11 | 80.00 | 91.75 |
| 44 | 0.4687 | 80.00 | 87.18 | 75.56 | 80.95 | 89.90 |
| 45 | 0.4256 | 82.50 | 94.29 | 73.33 | 82.50 | 91.75 |
| 46 | 0.4571 | 80.00 | 93.94 | 68.89 | 79.49 | 90.03 |
| 47 | 0.4416 | 82.50 | 91.89 | 75.56 | 82.93 | 91.17 |
| 48 | 0.4201 | 82.50 | 91.89 | 75.56 | 82.93 | 90.86 |
| 49 | 0.4388 | 85.00 | 92.31 | 80.00 | 85.71 | 89.21 |
| **50** ⭐ | **0.4386** | **86.25** | **92.50** | **82.22** | **87.06** | **88.57** |
| 51 | 0.4640 | 81.25 | 94.12 | 71.11 | 81.01 | 88.76 |
| 52 | 0.4422 | 77.50 | 90.91 | 66.67 | 76.92 | 87.43 |

> ⭐ **Epoch 50** is where the best model was saved (`best_model_ffpp.pth`) with peak validation accuracy of **86.25%**.

---

## Key Training Trends

### Loss Progression
- **Epoch 1**: Train Loss = 1.109 (starting point)
- **Epoch 8**: Train Loss = 0.718 (end of Phase 1)
- **Epoch 25**: Train Loss dropped below 0.50
- **Epoch 45**: Train Loss dropped below 0.43 (lowest recorded: ~0.420 at epoch 48)

### Accuracy Progression
- **Epochs 1–8** (Phase 1): Val Acc ranged between 50–63%, model learning basic features
- **Epochs 9–20** (Phase 2 early): Val Acc climbed to ~73%, significant improvement
- **Epochs 29–35**: Crossed the **80%** threshold consistently
- **Epoch 40**: First time above **83.75%**
- **Epoch 49–50**: Peak performance at **85–86.25%**

### AUC-ROC Progression
- Started at ~56% (epoch 1)
- Crossed **80%** at epoch 17
- Crossed **90%** at epoch 33
- Peaked at **~91.7%** around epochs 35–43

---

## Training Configuration
- **Total Epochs Planned**: 80 (ran 86 with restart/continuation)
- **Batch Size**: 4
- **Sequence Length**: 15 frames/video
- **Optimizer**: AdamW (LR=0.0001, weight_decay=0.0001)
- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.3)
- **Augmentation**: Mixup (alpha=0.2), face cropping, standard augmentations
- **AMP**: Mixed precision enabled
- **GPU**: NVIDIA GeForce RTX 3050 6GB Laptop GPU
