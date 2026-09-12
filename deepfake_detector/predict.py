"""
Deepfake Detector — Single Video Prediction
============================================
Run this script, choose a video file from the file picker,
and the model will instantly classify it as REAL or FAKE.

Usage:
    python predict.py                        # Opens file picker
    python predict.py --video path/to/vid.mp4  # Direct path
"""

import os
import sys
import argparse
import cv2
import numpy as np
import torch
from torchvision import transforms

import dataset_config as cfg
from model import Model
from data_loader_ffpp import detect_and_crop_face


# ─────────────────────────────────────────────────────────────────
# File picker using tkinter (built into Python — no install needed)
# ─────────────────────────────────────────────────────────────────
def pick_video_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()           # hide the empty root window
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(
            title="Select a Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files",   "*.*"),
            ]
        )
        root.destroy()
        return file_path
    except Exception as e:
        print(f"  [ERROR] Could not open file picker: {e}")
        print("  Please use:  python predict.py --video path/to/your/video.mp4")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────
def load_model(model_path, device):
    if not os.path.exists(model_path):
        print(f"\n  [ERROR] Model not found: {model_path}")
        print("  Make sure training is complete and models/best_model_ffpp.pth exists.")
        sys.exit(1)

    model = Model(
        num_classes=cfg.NUM_CLASSES,
        latent_dim=cfg.LATENT_DIM,
        lstm_layers=cfg.LSTM_LAYERS,
        hidden_dim=cfg.HIDDEN_DIM,
        bidirectional=cfg.BIDIRECTIONAL,
        dropout=cfg.DROPOUT,
    )

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get(
            'model_state_dict',
            checkpoint.get('state_dict', checkpoint)
        )
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────
# Video → tensor
# ─────────────────────────────────────────────────────────────────
def video_to_tensor(video_path, sequence_length=cfg.SEQUENCE_LENGTH):
    """Extract frames from a video, crop faces, return a model-ready tensor."""

    # ── Read all frames ──
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    all_frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        all_frames.append(frame)
    cap.release()

    total = len(all_frames)
    if total == 0:
        print("  [ERROR] No frames could be read from the video.")
        sys.exit(1)

    print(f"  Total frames in video : {total}")

    # ── Uniformly sample `sequence_length` frames ──
    indices = np.linspace(0, total - 1, sequence_length, dtype=int)
    sampled = [all_frames[i] for i in indices]

    # ── Face detection + transform ──
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.MEAN, cfg.STD),
    ])

    tensors = []
    faces_found = 0
    for frame in sampled:
        face = detect_and_crop_face(frame)
        if face is not None and face.shape[0] > 10 and face.shape[1] > 10:
            faces_found += 1
        frame_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        tensors.append(val_transform(frame_rgb))

    print(f"  Frames used           : {sequence_length}")
    print(f"  Faces detected        : {faces_found}/{sequence_length}")

    # Shape: (1, sequence_length, C, H, W)
    return torch.stack(tensors).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────
def predict(model, tensor, device):
    tensor = tensor.to(device)
    with torch.no_grad():
        if cfg.USE_AMP and device.type == 'cuda':
            from torch.cuda.amp import autocast
            with autocast():
                _, logits = model(tensor)
        else:
            _, logits = model(tensor)

    probs = torch.softmax(logits, dim=1).squeeze()   # shape: (2,)
    fake_prob = probs[cfg.LABEL_FAKE].item() * 100
    real_prob = probs[cfg.LABEL_REAL].item() * 100
    pred_class = torch.argmax(probs).item()
    label = "REAL" if pred_class == cfg.LABEL_REAL else "FAKE"
    confidence = real_prob if pred_class == cfg.LABEL_REAL else fake_prob
    return label, confidence, real_prob, fake_prob


# ─────────────────────────────────────────────────────────────────
# Pretty result printer
# ─────────────────────────────────────────────────────────────────
def print_result(video_path, label, confidence, real_prob, fake_prob):
    is_fake = label == "FAKE"

    # Visual indicator
    bar_len   = 40
    filled    = int(bar_len * confidence / 100)
    bar       = "█" * filled + "░" * (bar_len - filled)

    print("\n" + "=" * 60)
    print("  PREDICTION RESULT")
    print("=" * 60)
    print(f"  File   : {os.path.basename(video_path)}")
    print()

    if is_fake:
        print(f"  ██████  ✗  FAKE VIDEO DETECTED  ✗  ██████")
    else:
        print(f"  ██████  ✓  REAL VIDEO DETECTED  ✓  ██████")

    print()
    print(f"  Verdict    : {label}")
    print(f"  Confidence : {confidence:.1f}%")
    print()
    print(f"  REAL : {real_prob:5.1f}%  [{bar if not is_fake else '░'*bar_len}]")
    print(f"  FAKE : {fake_prob:5.1f}%  [{bar if is_fake else '░'*bar_len}]")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Deepfake single-video classifier")
    parser.add_argument(
        '--video',
        default=None,
        help="Path to a video file. If omitted, a file picker dialog opens."
    )
    parser.add_argument(
        '--model',
        default=cfg.BEST_MODEL_PATH,
        help=f"Path to model .pth (default: {cfg.BEST_MODEL_PATH})"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  DEEPFAKE DETECTOR — Single Video Mode")
    print("=" * 60)

    # ── Pick video ──
    if args.video:
        video_path = args.video
    else:
        print("\n  Opening file picker — choose a video file...")
        video_path = pick_video_file()

    if not video_path:
        print("  No file selected. Exiting.")
        sys.exit(0)

    if not os.path.exists(video_path):
        print(f"  [ERROR] File not found: {video_path}")
        sys.exit(1)

    print(f"\n  Video selected : {video_path}")

    # ── Device ──
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            device = torch.device("cuda")
            print(f"  Device         : GPU ({torch.cuda.get_device_name(0)})")
        except RuntimeError:
            device = torch.device("cpu")
            print("  Device         : CPU (CUDA error fallback)")
    else:
        device = torch.device("cpu")
        print("  Device         : CPU")

    # ── Load model ──
    print(f"  Model          : {args.model}")
    print("\n  Loading model...", end=" ", flush=True)
    model = load_model(args.model, device)
    print("done ✓")

    # ── Process video ──
    print("\n  Processing video...")
    tensor = video_to_tensor(video_path)

    # ── Predict ──
    print("  Running inference...", end=" ", flush=True)
    label, confidence, real_prob, fake_prob = predict(model, tensor, device)
    print("done ✓")

    # ── Show result ──
    print_result(video_path, label, confidence, real_prob, fake_prob)


if __name__ == "__main__":
    main()
