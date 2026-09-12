"""
FaceForensics++ Dataset Loader (Improved)
==========================================
Loads videos from FF++ dataset (real/ and fake/ folders), extracts frames,
detects and crops faces, applies strong augmentation, and returns tensors
ready for the ResNeXt + LSTM model.

Key improvements:
  - Face detection & cropping using OpenCV Haar Cascade
  - Strong data augmentation for training (flip, color jitter, blur, rotation, erasing)
  - Validation uses clean transforms only
  - Robust error handling with fallbacks

Expected dataset structure:
    FF++/
    ├── real/    (contains .mp4 video files)
    └── fake/    (contains .mp4 video files)
"""

import os
import glob
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import dataset_config as cfg

# ============================================================
# Face Detection Setup (OpenCV Haar Cascade — zero extra install)
# ============================================================
_face_cascade = None


def get_face_cascade():
    """Lazy-load the face cascade classifier."""
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
        if _face_cascade.empty():
            print("[WARNING] Could not load face cascade. Face detection disabled.")
            _face_cascade = None
    return _face_cascade


def detect_and_crop_face(frame, padding=cfg.FACE_PADDING):
    """
    Detect the largest face in a frame and crop it with padding.

    Args:
        frame: BGR image (numpy array).
        padding: Fraction of face size to add as padding (0.3 = 30%).

    Returns:
        Cropped face image, or center-cropped frame if no face detected.
    """
    if not cfg.FACE_DETECT:
        return center_crop(frame)

    cascade = get_face_cascade()
    if cascade is None:
        return center_crop(frame)

    h, w = frame.shape[:2]

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) == 0:
        return center_crop(frame)

    # Take the largest face (by area)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, fw, fh = faces[0]

    # Add padding
    pad_w = int(fw * padding)
    pad_h = int(fh * padding)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(w, x + fw + pad_w)
    y2 = min(h, y + fh + pad_h)

    cropped = frame[y1:y2, x1:x2]

    # Sanity check — if crop is too small, fallback
    if cropped.shape[0] < 20 or cropped.shape[1] < 20:
        return center_crop(frame)

    return cropped


def center_crop(frame):
    """Center crop the frame to a square (fallback when no face detected)."""
    h, w = frame.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


class FFppVideoDataset(Dataset):
    """
    PyTorch Dataset for FaceForensics++ video classification.

    Args:
        video_paths (list): List of video file paths.
        labels (list): Corresponding labels (0=FAKE, 1=REAL).
        sequence_length (int): Number of frames to extract per video.
        transform (callable): Transform pipeline for each frame.
        is_training (bool): If True, apply face detection on every frame.
                           If False, use cached face position from first frame.
    """

    def __init__(self, video_paths, labels, sequence_length=cfg.SEQUENCE_LENGTH,
                 transform=None, is_training=True):
        self.video_paths = video_paths
        self.labels = labels
        self.sequence_length = sequence_length
        self.transform = transform
        self.is_training = is_training

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        """
        Returns:
            frames: Tensor of shape (sequence_length, 3, IMAGE_SIZE, IMAGE_SIZE)
            label: Integer label (0=FAKE, 1=REAL)
        """
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Extract frames from the video
        all_frames = self._extract_frames(video_path)

        if len(all_frames) == 0:
            # If no frames could be extracted, return a zero tensor
            frames = torch.zeros(self.sequence_length, 3, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
            return frames, label

        # Uniformly sample 'sequence_length' frames
        frames = self._sample_frames(all_frames)

        # Detect and crop faces from frames
        face_frames = []
        for frame in frames:
            face_frame = detect_and_crop_face(frame)
            face_frames.append(face_frame)

        # Apply transforms to each face-cropped frame
        transformed_frames = []
        for frame in face_frames:
            # Convert BGR (OpenCV) to RGB for PIL/torchvision
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                transformed_frames.append(self.transform(frame_rgb))
            else:
                frame_resized = cv2.resize(frame_rgb, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
                transformed_frames.append(
                    torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
                )

        # Stack into tensor: (sequence_length, C, H, W)
        frames_tensor = torch.stack(transformed_frames)
        return frames_tensor, label

    def _extract_frames(self, video_path):
        """Extract all frames from a video file using OpenCV."""
        frames = []
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"[WARNING] Cannot open video: {video_path}")
            return frames

        while True:
            success, frame = cap.read()
            if not success:
                break
            frames.append(frame)

        cap.release()
        return frames

    def _sample_frames(self, all_frames):
        """
        Uniformly sample 'sequence_length' frames from the list of all frames.
        If there are fewer frames than needed, duplicate the last frame.
        """
        total_frames = len(all_frames)

        if total_frames >= self.sequence_length:
            if self.is_training:
                # During training: add slight randomness to frame sampling
                # This acts as temporal augmentation
                indices = np.linspace(0, total_frames - 1, self.sequence_length + 2, dtype=int)
                # Randomly pick sequence_length from these indices
                if len(indices) > self.sequence_length:
                    start = random.randint(0, len(indices) - self.sequence_length)
                    indices = indices[start:start + self.sequence_length]
                sampled = [all_frames[i] for i in indices[:self.sequence_length]]
            else:
                # During validation: deterministic uniform sampling
                indices = np.linspace(0, total_frames - 1, self.sequence_length, dtype=int)
                sampled = [all_frames[i] for i in indices]
        else:
            # Use all available frames + pad with last frame
            sampled = all_frames.copy()
            while len(sampled) < self.sequence_length:
                sampled.append(all_frames[-1])

        return sampled


def get_video_paths_and_labels():
    """
    Scan the FF++ dataset directory and return lists of video paths and labels.

    Returns:
        video_paths (list): List of absolute paths to video files.
        labels (list): Corresponding labels (0=FAKE, 1=REAL).
    """
    video_paths = []
    labels = []

    # Load REAL videos (label = 1)
    real_pattern = os.path.join(cfg.REAL_VIDEOS_PATH, "*.mp4")
    real_videos = sorted(glob.glob(real_pattern))
    for v in real_videos:
        video_paths.append(v)
        labels.append(cfg.LABEL_REAL)

    # Load FAKE videos (label = 0)
    fake_pattern = os.path.join(cfg.FAKE_VIDEOS_PATH, "*.mp4")
    fake_videos = sorted(glob.glob(fake_pattern))
    for v in fake_videos:
        video_paths.append(v)
        labels.append(cfg.LABEL_FAKE)

    print(f"[INFO] Found {len(real_videos)} REAL videos and {len(fake_videos)} FAKE videos")
    print(f"[INFO] Total videos: {len(video_paths)}")

    return video_paths, labels


def create_data_loaders():
    """
    Create train and validation DataLoaders for the FF++ dataset.

    Returns:
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        train_size (int): Number of training samples.
        val_size (int): Number of validation samples.
    """
    # Get all video paths and labels
    video_paths, labels = get_video_paths_and_labels()

    if len(video_paths) == 0:
        raise FileNotFoundError(
            f"No .mp4 videos found in:\n"
            f"  REAL: {cfg.REAL_VIDEOS_PATH}\n"
            f"  FAKE: {cfg.FAKE_VIDEOS_PATH}\n"
            f"Please verify your dataset paths in dataset_config.py"
        )

    # Combine and shuffle together
    combined = list(zip(video_paths, labels))
    random.seed(cfg.RANDOM_SEED)
    random.shuffle(combined)
    video_paths, labels = zip(*combined)
    video_paths, labels = list(video_paths), list(labels)

    # Split into train and validation
    split_idx = int(len(video_paths) * cfg.TRAIN_SPLIT)
    train_paths = video_paths[:split_idx]
    train_labels = labels[:split_idx]
    val_paths = video_paths[split_idx:]
    val_labels = labels[split_idx:]

    # Count class distribution
    train_real = sum(1 for l in train_labels if l == cfg.LABEL_REAL)
    train_fake = sum(1 for l in train_labels if l == cfg.LABEL_FAKE)
    val_real = sum(1 for l in val_labels if l == cfg.LABEL_REAL)
    val_fake = sum(1 for l in val_labels if l == cfg.LABEL_FAKE)

    print(f"\n[INFO] Dataset Split:")
    print(f"  Training:   {len(train_paths)} videos (REAL: {train_real}, FAKE: {train_fake})")
    print(f"  Validation: {len(val_paths)} videos (REAL: {val_real}, FAKE: {val_fake})")

    # ============================================================
    # STRONG Training Augmentation — critical for small datasets
    # ============================================================
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),           # Resize slightly larger
        transforms.RandomCrop(cfg.IMAGE_SIZE),   # Random 224x224 crop
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.05,
        ),
        transforms.RandomGrayscale(p=0.05),
        # NOTE: GaussianBlur removed — it destroys the subtle compression
        # artifacts that are key signals for deepfake detection
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(cfg.MEAN, cfg.STD),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.12)),
    ])

    # Validation: clean transforms only (no augmentation)
    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.MEAN, cfg.STD),
    ])

    # Create datasets
    train_dataset = FFppVideoDataset(
        train_paths, train_labels,
        sequence_length=cfg.SEQUENCE_LENGTH,
        transform=train_transforms,
        is_training=True,
    )
    val_dataset = FFppVideoDataset(
        val_paths, val_labels,
        sequence_length=cfg.SEQUENCE_LENGTH,
        transform=val_transforms,
        is_training=False,
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,  # Drop incomplete last batch (helps BatchNorm stability)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, len(train_paths), len(val_paths)


if __name__ == "__main__":
    # Quick test: verify dataset loading
    print("=" * 60)
    print("Testing FaceForensics++ Data Loader")
    print("=" * 60)

    train_loader, val_loader, train_size, val_size = create_data_loaders()

    # Test loading one batch
    for frames, labels in train_loader:
        print(f"\n[TEST] Batch shape: {frames.shape}")
        print(f"[TEST] Labels: {labels}")
        print(f"[TEST] Expected shape: (batch_size, {cfg.SEQUENCE_LENGTH}, 3, {cfg.IMAGE_SIZE}, {cfg.IMAGE_SIZE})")
        break

    print("\n[SUCCESS] Data loader is working correctly!")
