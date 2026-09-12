"""
Dataset Configuration for FaceForensics++ (FF++) Training Pipeline
=================================================================
All configurable paths, hyperparameters, and settings are defined here.
Modify this file to match your local setup before running train.py.
"""

import os

# ============================================================
# Dataset Paths
# ============================================================
# Root directory of the FF++ dataset (contains 'real/' and 'fake/' subfolders with .mp4 videos)
DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FF++")

# Paths to real and fake video folders
REAL_VIDEOS_PATH = os.path.join(DATASET_ROOT, "real")
FAKE_VIDEOS_PATH = os.path.join(DATASET_ROOT, "fake")

# ============================================================
# Train / Validation Split
# ============================================================
TRAIN_SPLIT = 0.8   # 80% training, 20% validation
RANDOM_SEED = 42     # For reproducibility

# ============================================================
# Frame Extraction Settings
# ============================================================
# Number of frames to extract per video
SEQUENCE_LENGTH = 15

# Minimum number of frames a video must have to be included
MIN_FRAMES_REQUIRED = 20

# ============================================================
# Face Detection
# ============================================================
FACE_DETECT = True   # Enable face cropping from frames
FACE_PADDING = 0.3   # Padding around detected face (30%)

# ============================================================
# Image Processing
# ============================================================
# ResNeXt50 was pretrained on 224x224 — matching this is critical
IMAGE_SIZE = 224

# ImageNet normalization (used by pretrained ResNeXt)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ============================================================
# Training Hyperparameters
# ============================================================
BATCH_SIZE = 4
NUM_EPOCHS = 95  # Resume from epoch 81 → train 15 more epochs (81-95)
NUM_WORKERS = 2      # DataLoader workers (reduce to 0 if issues on Windows)

# --- Two-Phase Learning Rate Strategy ---
# Phase 1 (epochs 1 to UNFREEZE_EPOCH): Backbone frozen, train LSTM + head only
PHASE1_LR = 1e-4
# Phase 2 (epochs UNFREEZE_EPOCH+1 to NUM_EPOCHS): Unfreeze last backbone blocks
PHASE2_LR = 1e-5
LEARNING_RATE = PHASE1_LR  # Starting LR (Phase 1)

WEIGHT_DECAY = 1e-4
UNFREEZE_EPOCH = 8   # Epoch at which to unfreeze backbone (EfficientNet converges faster)

# --- Scheduler ---
SCHEDULER_PATIENCE = 5   # ReduceLROnPlateau patience (more patient to avoid premature LR decay)
SCHEDULER_FACTOR = 0.3   # LR reduction factor (cut more aggressively when we do reduce)

# --- Gradient Clipping ---
GRADIENT_CLIP = 1.0

# --- Mixed Precision ---
USE_AMP = True  # Automatic Mixed Precision (saves VRAM on RTX 3050 6GB)

# ============================================================
# Model Architecture
# ============================================================
NUM_CLASSES = 2       # REAL vs FAKE
LATENT_DIM = 1536     # EfficientNet-B3 output feature dimension
LSTM_LAYERS = 2       # 2 LSTM layers for better temporal modeling
HIDDEN_DIM = 256      # Smaller hidden dim = less overfitting + less VRAM
BIDIRECTIONAL = True   # Bidirectional LSTM for richer temporal context
DROPOUT = 0.4          # Slightly less dropout (EfficientNet is already smaller)

# ============================================================
# Model Saving
# ============================================================
MODEL_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "best_model_ffpp.pth")
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_DIR, "checkpoint_ffpp.pth")

# ============================================================
# Labels
# ============================================================
# Label encoding: FAKE = 0, REAL = 1
LABEL_FAKE = 0
LABEL_REAL = 1
