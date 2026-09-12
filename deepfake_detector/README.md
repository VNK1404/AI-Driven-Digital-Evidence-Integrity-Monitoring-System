# 🎭 Deepfake Detection using Deep Learning
### EfficientNet-B3 + Bidirectional LSTM + Temporal Attention

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

---



## 📖 Introduction

This project implements a complete **deepfake video detection** system through a hybrid deep learning methodology. It integrates the spatial feature learning capability of **EfficientNet-B3** (a state-of-the-art CNN backbone) with the temporal pattern modeling of a **Bidirectional LSTM** further enriched by a **Temporal Attention** mechanism.

The model accepts sequences of face-cropped video frames, leverages EfficientNet-B3 to capture spatial forgery indicators frame by frame, uses BiLSTM to model temporal anomalies across frames, and applies attention to zero in on the most revealing frames — achieving a **best validation accuracy of 86.25%** on the FaceForensics++ dataset.

---

## 🏗️ System Architecture

<p align="center">
  <img src="github_assets/System Architecture.png" alt="System Architecture" width="80%"/>
</p>

The detection workflow passes through three distinct stages:
1. **Frame Extraction** — Sample `N` frames at uniform intervals from each video
2. **Face Detection & Cropping** — Detect and crop facial areas using OpenCV
3. **Classification** — EfficientNet-B3 extracts per-frame features → BiLSTM models temporal context → Attention pooling → FC head predicts REAL/FAKE

---
## 🔍 Grad‑CAM Explainable AI

We have integrated Gradient‑weighted Class Activation Mapping (Grad‑CAM) into the Django web interface. When running inference, the model:

- Enables gradient flow through the final convolutional layer of EfficientNet‑B3.
- Derives the gradient of the class prediction score with respect to that layer's feature map.
- Generates a heatmap that pinpoints facial regions (eyes, nose, mouth) most decisive for the classification.
- Composites the heatmap with the original face crop (60 % face, 40 % heatmap) and writes the resulting images to `django_app/uploaded_images/`.

The **Predict** page now displays a **"Grad‑CAM Activation Mapping (Where AI Looked)"** section featuring blended heatmaps for each sampled frame, making the model's decision process visually transparent and understandable.
## 🧠 Model Architecture

```
Input: (B, T, 3, 224, 224)     ← B videos, T=15 frames each
         │
    ┌────▼────────────────────────────────────────────────┐
    │  EfficientNet-B3 (pretrained on ImageNet)            │
    │  ├── Early blocks (frozen always)                    │
    │  ├── Mid blocks   (unfrozen in Phase 2)              │
    │  └── Late blocks  (unfrozen in Phase 2)              │
    │  Output: (B×T, 1536)                                 │
    └──────────────────────────────────────────────────────┘
         │
    ┌────▼────────────────────────────────────────────────┐
    │  Bidirectional LSTM (2 layers, hidden=256)           │
    │  Output: (B, T, 512)                                 │
    └──────────────────────────────────────────────────────┘
         │
    ┌────▼────────────────────────────────────────────────┐
    │  Temporal Attention                                  │
    │  Learns which frames are most suspicious             │
    │  Output: (B, 512)                                    │
    └──────────────────────────────────────────────────────┘
         │
    ┌────▼────────────────────────────────────────────────┐
    │  FC Head: 512 → 128 → 2 (REAL / FAKE)               │
    │  LayerNorm + LeakyReLU + Dropout                     │
    └──────────────────────────────────────────────────────┘
```

### Key Design Choices
| Component | Choice | Reason |
|---|---|---|
| CNN Backbone | EfficientNet-B3 | 82% ImageNet top-1, ~12M params (less overfitting than ResNeXt50) |
| Temporal Model | Bidirectional LSTM | Captures both past and future frame context |
| Attention | Temporal Attention | Automatically focuses on most manipulated frames |
| Training | Two-Phase (Freeze → Unfreeze) | Prevents catastrophic forgetting of ImageNet features |
| Augmentation | Mixup (α=0.2) | Significantly improves generalization |
| Precision | AMP (Mixed Precision) | Saves VRAM on consumer GPUs |

---

## 📊 Results

| Metric | Value |
|---|---|
| **Best Validation Accuracy** | **86.25%** (Epoch 50) |
| Best Precision | 92.50% |
| Best Recall | 82.22% |
| Best F1-Score | 87.06% |
| Best AUC-ROC | ~91.75% |
| Total Epochs Trained | 86 |

### Training Progression

| Phase | Epochs | Backbone | Val Acc Range |
|---|---|---|---|
| Phase 1 | 1–8 | Frozen | 50–63% |
| Phase 2 (early) | 9–20 | Partially Unfrozen | 61–73% |
| Phase 2 (mid) | 21–40 | Partially Unfrozen | 73–83% |
| Phase 2 (best) | 41–50 | Partially Unfrozen | 82–**86.25%** |

For complete per-epoch metrics, see [`training_metrics_report.md`](training_metrics_report.md).

---

## 📁 Project Structure

```
deepfake-detection/
│
├── model.py                    # EfficientNet-B3 + BiLSTM + Attention model definition
├── dataset_config.py           # All hyperparameters and config (edit before training!)
├── data_loader_ffpp.py         # Dataset class, frame extraction, augmentations
├── train.py                    # Full training pipeline (2-phase, AMP, checkpointing)
├── evaluate.py                 # Evaluation: accuracy, precision, recall, F1, AUC-ROC
├── predict.py                  # Inference on new videos
├── utils.py                    # Checkpoint save/load, EarlyStopping, utilities
├── requirements.txt            # Python dependencies
├── training_metrics_report.md  # Detailed per-epoch results
├── .gitignore
│
├── Model Creation/             # Jupyter notebooks for experimentation
│   ├── preprocessing.ipynb     # Frame extraction and face cropping
│   ├── Model_and_train_csv.ipynb  # Model training (notebook version)
│   ├── Predict.ipynb           # Prediction notebook
│   ├── Readme.md
│   ├── labels/
│   │   └── Gobal_metadata.csv  # Video metadata and labels
│   └── Helpers/
│       ├── copy real and fake .ipynb
│       ├── Create_csv_from_glob.ipynb
│       ├── deepfake-starter-kit.ipynb
│       ├── for_Balancing_data.ipynb
│       ├── label_json_to_csv.py
│       └── Remove_audio_altered_files.ipynb
│
└── github_assets/
    ├── System Architecture.png
    └── fakegif.gif
```

---

## 🗄️ Dataset

This project uses the **[FaceForensics++](https://github.com/ondyari/FaceForensics)** dataset.

> ⚠️ **The dataset is NOT included in this repository** due to its large size and licensing requirements.

### How to get FaceForensics++

1. Access the official repository: [https://github.com/ondyari/FaceForensics](https://github.com/ondyari/FaceForensics)
2. Fill in and submit the **access request form** on the FaceForensics++ page  
3. Use the download script provided to obtain the dataset
4. Arrange the video files according to the directory layout shown below:

```
FF++/
├── real/        ← Real face videos (.mp4)
└── fake/        ← DeepFake manipulated videos (.mp4)
```

5. Place the `FF++/` folder in the project root (same level as `train.py`)

> The FaceForensics++ dataset contains ~1000 original videos and their manipulated versions using methods like:
> - **DeepFakes** (face-swap using autoencoders)
> - **Face2Face** (expression transfer)
> - **FaceSwap** (3D model-based face replacement)
> - **NeuralTextures** (neural rendering-based)

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA 12.1 (recommended: 6GB+ VRAM)
- CUDA Toolkit installed

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/deepfake-detection.git
cd deepfake-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install PyTorch with CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1: Configure Your Dataset Path

Open `dataset_config.py` and update the dataset path along with any desired hyperparameters:

```python
# dataset_config.py
DATASET_ROOT = "path/to/your/FF++"   # Update this!
SEQUENCE_LENGTH = 15                  # Frames per video
BATCH_SIZE = 4                        # Reduce if OOM
NUM_EPOCHS = 80
```

### Step 2: Train the Model

```bash
python train.py
```

The training process is split into **two phases**:
- **Phase 1** (Epochs 1–8): The EfficientNet backbone is locked; training focuses on the LSTM, attention, and FC classifier layers.
- **Phase 2** (Epoch 9+): Mid and late backbone layers are unlocked for fine-tuning at a significantly reduced learning rate.

After each epoch, checkpoints are saved to `models/` automatically. Interrupted training sessions resume from the most recent checkpoint.

### Step 3: Evaluate the Model

```bash
python evaluate.py
```

Outputs accuracy, precision, recall, F1-score, and AUC-ROC scores computed on the validation set.

### Step 4: Predict on a New Video

```bash
python predict.py --video path/to/video.mp4
```

---

## 📋 Configuration Reference

All settings are in `dataset_config.py`:

| Parameter | Default | Description |
|---|---|---|
| `SEQUENCE_LENGTH` | 15 | Frames sampled per video |
| `BATCH_SIZE` | 4 | Training batch size |
| `NUM_EPOCHS` | 95 | Total training epochs |
| `PHASE1_LR` | 1e-4 | Learning rate for Phase 1 |
| `PHASE2_LR` | 1e-5 | Learning rate for Phase 2 fine-tuning |
| `UNFREEZE_EPOCH` | 8 | Epoch at which backbone is unfrozen |
| `HIDDEN_DIM` | 256 | LSTM hidden state size |
| `LSTM_LAYERS` | 2 | Number of LSTM layers |
| `DROPOUT` | 0.4 | Dropout rate |
| `USE_AMP` | True | Mixed Precision Training |
| `FACE_DETECT` | True | Enable face detection/cropping |

---

## 🔬 Training Details

- **Optimizer**: AdamW with weight decay 1e-4
- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.3)
- **Augmentation**: Random horizontal flip, color jitter, random grayscale, Mixup (α=0.2)
- **Face Detection**: OpenCV Haar Cascade with 30% padding
- **Mixed Precision**: Enabled via `torch.cuda.amp`
- **GPU Used**: NVIDIA GeForce RTX 3050 6GB

---

## 🤝 Contributing

Contributions are welcome! Here are some ideas:
- [ ] Improve detection accuracy beyond 90%
- [ ] Add support for other deepfake datasets (Celeb-DF, DFDC)
- [ ] Create a web demo / REST API for inference
- [ ] Add Grad-CAM visualization for explainability
- [ ] Optimize inference for real-time detection

---

## 📄 License

This project is licensed under the **GNU General Public License v3.0**.  
See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [FaceForensics++](https://github.com/ondyari/FaceForensics) by Andreas Rössler et al. for the dataset
- [EfficientNet](https://arxiv.org/abs/1905.11946) by Tan & Le (Google Brain)
- [PyTorch](https://pytorch.org/) for the deep learning framework
- Original ResNeXt+LSTM baseline inspired by [abhijitjadhav1998](https://github.com/abhijitjadhav1998/Deepfake_detection_using_deep_learning)
