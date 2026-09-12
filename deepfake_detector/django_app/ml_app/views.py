"""
views.py — Core inference logic for Deepfake Detection Web App
==============================================================
Model: EfficientNet-B3 + Bidirectional LSTM + Temporal Attention
Dataset: FaceForensics++ (FF++)

Adapted from original Django app to work with the new EfficientNet-B3 + BiLSTM architecture.
The model file (best_model_ffpp.pth) must be placed in the 'models/' directory.
"""

import os
import sys
import time
import shutil

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image as PImage
from django.shortcuts import render, redirect
from django.conf import settings

from .forms import VideoUploadForm

# ─── Add parent directory to path so we can import model/dataset_config ───────
# This allows importing model.py and dataset_config.py from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ─── Import model and config from the main project ────────────────────────────
try:
    import dataset_config as cfg
    from model import Model
    MODEL_IMPORTED = True
except ImportError:
    MODEL_IMPORTED = False
    print("[WARNING] Could not import model.py or dataset_config.py from the project root.")
    print("          Make sure django_app/ is inside the same folder as model.py")

# ─── Constants ────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 15          # Must match training config (cfg.SEQUENCE_LENGTH)
IMAGE_SIZE      = 224         # EfficientNet-B3 input size
MEAN            = [0.485, 0.456, 0.406]
STD             = [0.229, 0.224, 0.225]
FACE_PADDING    = 0.20        # 20% padding around detected face
ALLOWED_EXTS    = {'mp4', 'avi', 'mov', 'webm', 'mkv', 'flv', '3gp', 'wmv'}

# ─── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {DEVICE}")

# ─── Image pre-processing pipeline ───────────────────────────────────────────
val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ─── OpenCV Haar Cascade face detector ────────────────────────────────────────
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


# ─── Dataset ──────────────────────────────────────────────────────────────────
class VideoDataset(Dataset):
    """
    Loads a list of video file paths.
    For each video, extracts SEQUENCE_LENGTH evenly-spaced frames,
    detects and crops the face from each frame, and applies val_transforms.
    """

    def __init__(self, video_paths, sequence_length=SEQUENCE_LENGTH, transform=None):
        self.video_paths     = video_paths
        self.sequence_length = sequence_length
        self.transform       = transform or val_transforms

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        path   = self.video_paths[idx]
        frames = self._extract_frames(path)
        tensor = torch.stack(frames)          # (T, C, H, W)
        return tensor.unsqueeze(0)            # (1, T, C, H, W)

    def _extract_frames(self, path):
        cap      = cv2.VideoCapture(path)
        total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices  = np.linspace(0, max(total - 1, 0), self.sequence_length, dtype=int)
        frames   = []

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if not ret:
                # Pad with last frame if read fails
                if frames:
                    frames.append(frames[-1])
                continue

            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face  = self._crop_face(rgb)
            frames.append(self.transform(face))

        cap.release()

        # Pad with last valid frame if we got fewer than needed
        while len(frames) < self.sequence_length:
            frames.append(frames[-1] if frames else torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE))

        return frames[:self.sequence_length]

    def _crop_face(self, rgb_frame):
        """Detect and crop face from an RGB frame using Haar Cascade."""
        gray  = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        if len(faces) > 0:
            x, y, w, h = faces[0]
            pad_x = int(w * FACE_PADDING)
            pad_y = int(h * FACE_PADDING)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(rgb_frame.shape[1], x + w + pad_x)
            y2 = min(rgb_frame.shape[0], y + h + pad_y)
            return rgb_frame[y1:y2, x1:x2]

        # No face found — return original frame
        return rgb_frame


# ─── Model Loader ─────────────────────────────────────────────────────────────
_model_cache = None   # Cache to avoid reloading on each request

def load_model():
    """Load the EfficientNet-B3 + BiLSTM model from disk (cached)."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    if not MODEL_IMPORTED:
        raise RuntimeError(
            "Cannot load model: model.py or dataset_config.py not found. "
            "Make sure django_app/ is inside the project root next to model.py."
        )

    model_path = settings.MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at: {model_path}\n"
            "Please copy best_model_ffpp.pth into the django_app/models/ directory."
        )

    model = Model().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Support both raw state_dict saves and checkpoint dict saves
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    _model_cache = model
    print(f"[INFO] Model loaded from {model_path}")
    return model


# ─── Inference ────────────────────────────────────────────────────────────────
def run_inference(model, video_path):
    """
    Run deepfake detection on a single video file.

    Returns:
        dict with keys: label (str), confidence (float 0-100), label_index (int)
    """
    dataset    = VideoDataset([video_path], sequence_length=SEQUENCE_LENGTH)
    frames_tensor = dataset[0]              # (1, T, C, H, W)
    frames_tensor = frames_tensor.to(DEVICE)

    with torch.no_grad():
        _, logits = model(frames_tensor)
        probs     = torch.softmax(logits, dim=1)
        pred_idx  = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item() * 100.0

    label = "REAL" if pred_idx == 1 else "FAKE"
    return {"label": label, "confidence": round(confidence, 1), "label_index": pred_idx}


def generate_gradcam(model, frames_tensor, target_class=None):
    """
    Generates Grad-CAM heatmaps for the final convolutional layer of EfficientNet-B3.
    """
    # Switch to training mode temporarily for gradient computation on RNN layers (required by cuDNN)
    model.train()

    # Disable gradient tracking for parameters to save memory
    for param in model.parameters():
        param.requires_grad = False

    activations = None
    gradients = None

    def save_activation(module, input, output):
        nonlocal activations
        activations = output

    def save_gradient(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    # Target the late convolutional backbone sequential block
    target_layer = model.backbone_late

    handle_act = target_layer.register_forward_hook(save_activation)
    handle_grad = target_layer.register_full_backward_hook(save_gradient)

    with torch.set_grad_enabled(True):
        frames_tensor = frames_tensor.clone().detach().requires_grad_(True)
        # Forward pass in training mode to allow gradient flow through LSTM
        fmap, logits = model(frames_tensor)

        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred_idx].item() * 100.0

        if target_class is None:
            target_class = pred_idx

        model.zero_grad()
        score = logits[0, target_class]
        score.backward()

    # Clean hooks
    # Restore model to eval mode after Grad-CAM generation
    model.eval()
    handle_act.remove()
    handle_grad.remove()

    if gradients is None or activations is None:
        return None, pred_idx, confidence

    # Extract tensors
    act_data = activations.detach().cpu()   # shape: (15, 1536, H_f, W_f)
    grad_data = gradients.detach().cpu()    # shape: (15, 1536, H_f, W_f)

    # Calculate average gradients per channel
    weights = torch.mean(grad_data, dim=(2, 3), keepdim=True)  # (15, 1536, 1, 1)

    # Weighted combination of channels
    cam = torch.sum(weights * act_data, dim=1)  # (15, H_f, W_f)
    cam = torch.clamp(cam, min=0)  # ReLU

    heatmaps = []
    T = frames_tensor.shape[1]
    for t in range(T):
        frame_cam = cam[t].numpy()
        max_val = np.max(frame_cam)
        if max_val > 0:
            frame_cam = frame_cam / max_val
        heatmap_uint8 = np.uint8(255 * frame_cam)
        heatmaps.append(heatmap_uint8)

    return heatmaps, pred_idx, confidence


# ─── Helper: save preview frames ──────────────────────────────────────────────
def extract_preview_frames(video_path, video_name_only, num_frames=6):
    """
    Extract a few evenly-spaced frames from the video for display on the results page.
    Returns list of saved image filenames (relative, served via STATICFILES_DIRS).
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx   = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)

    saved = []
    for i, frame_no in enumerate(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
        ret, frame = cap.read()
        if not ret:
            continue
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img      = PImage.fromarray(rgb)
        fname    = f"{video_name_only}_preview_{i+1}.png"
        img.save(os.path.join(settings.PROJECT_DIR, 'uploaded_images', fname))
        saved.append(fname)

    cap.release()
    return saved


def extract_face_frames(video_path, video_name_only, num_frames=6):
    """
    Extract a few evenly-spaced face-cropped frames for display.
    Returns list of saved filenames (relative).
    """
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx   = np.linspace(0, max(total - 1, 0), num_frames, dtype=int)

    saved = []
    for i, frame_no in enumerate(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
        ret, frame = cap.read()
        if not ret:
            continue

        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        if len(faces) == 0:
            continue

        x, y, w, h = faces[0]
        pad_x = int(w * FACE_PADDING)
        pad_y = int(h * FACE_PADDING)
        x1 = max(0, x - pad_x);  y1 = max(0, y - pad_y)
        x2 = min(rgb.shape[1], x + w + pad_x);  y2 = min(rgb.shape[0], y + h + pad_y)
        face_crop = rgb[y1:y2, x1:x2]

        img   = PImage.fromarray(face_crop)
        fname = f"{video_name_only}_face_{i+1}.png"
        img.save(os.path.join(settings.PROJECT_DIR, 'uploaded_images', fname))
        saved.append(fname)

    cap.release()
    return saved


# ─── Utilities ────────────────────────────────────────────────────────────────
def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTS


# ─── Views ────────────────────────────────────────────────────────────────────
def index(request):
    """Home page: renders video upload form."""
    if request.method == 'GET':
        form = VideoUploadForm()
        # Clean up stale session data
        for key in ('file_name', 'prediction'):
            request.session.pop(key, None)
        return render(request, 'index.html', {'form': form})

    # POST — handle video upload
    form = VideoUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        return render(request, 'index.html', {'form': form})

    video_file = form.cleaned_data['upload_video_file']

    # Validate extension
    if not allowed_video_file(video_file.name):
        form.add_error('upload_video_file', 'Only video files are allowed (mp4, avi, mov, etc.)')
        return render(request, 'index.html', {'form': form})

    # Validate size
    if video_file.size > int(settings.MAX_UPLOAD_SIZE):
        form.add_error('upload_video_file', 'Maximum upload size is 100 MB.')
        return render(request, 'index.html', {'form': form})

    # Save the uploaded file
    ext         = video_file.name.split('.')[-1]
    saved_name  = f"upload_{int(time.time())}.{ext}"
    save_path   = os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_name)
    with open(save_path, 'wb') as f:
        shutil.copyfileobj(video_file, f)

    request.session['file_name'] = save_path
    return redirect('ml_app:predict')


def predict_page(request):
    """Prediction page: loads model, runs inference, shows results."""
    if 'file_name' not in request.session:
        return redirect('ml_app:home')

    video_path      = request.session['file_name']
    video_basename  = os.path.basename(video_path)
    video_name_only = os.path.splitext(video_basename)[0]

    try:
        # Load model (cached after first load)
        model = load_model()

        # Run inference and generate Grad-CAM
        start = time.time()
        dataset = VideoDataset([video_path], sequence_length=SEQUENCE_LENGTH)
        frames_tensor = dataset[0]              # (1, T, C, H, W)
        frames_tensor = frames_tensor.to(DEVICE)

        heatmaps, pred_idx, confidence = generate_gradcam(model, frames_tensor)
        elapsed = round(time.time() - start, 1)

        # Map predictions
        label = "REAL" if pred_idx == 1 else "FAKE"
        confidence = round(confidence, 1)

        # Extract preview frames for display
        preview_frames = extract_preview_frames(video_path, video_name_only, num_frames=8)
        face_frames    = extract_face_frames(video_path, video_name_only, num_frames=8)

        # Process and save Grad-CAM heatmaps
        gradcam_frames = []
        if heatmaps is not None:
            # Un-normalize the face crops from frames_tensor to get the original face images
            mean = torch.tensor(MEAN).view(3, 1, 1).to(frames_tensor.device)
            std = torch.tensor(STD).view(3, 1, 1).to(frames_tensor.device)

            unnormalized = frames_tensor[0] * std + mean
            unnormalized = torch.clamp(unnormalized, 0, 1)

            # Sample 8 frames evenly to match the face_frames previews
            num_frames = 8
            sampled_indices = np.linspace(0, SEQUENCE_LENGTH - 1, num_frames, dtype=int)

            for i, idx in enumerate(sampled_indices):
                # Convert tensor to numpy RGB
                face_tensor = unnormalized[idx].cpu()
                face_np = face_tensor.permute(1, 2, 0).numpy()
                face_np = np.uint8(255 * face_np)  # shape: (224, 224, 3)

                # Get and resize heatmap
                heatmap = heatmaps[idx]
                heatmap_resized = cv2.resize(heatmap, (224, 224))

                # Color-map heatmap (cv2.applyColorMap returns BGR, convert to RGB)
                heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

                # Overlay heatmap on original face (60% face, 40% heatmap)
                overlay = cv2.addWeighted(face_np, 0.6, heatmap_color_rgb, 0.4, 0)

                # Save blended overlay
                img = PImage.fromarray(overlay)
                fname = f"{video_name_only}_gradcam_{i+1}.png"
                img.save(os.path.join(settings.PROJECT_DIR, 'uploaded_images', fname))
                gradcam_frames.append(fname)

        context = {
            'output':           label,
            'confidence':       confidence,
            'preview_frames':   preview_frames,
            'face_frames':      face_frames,
            'gradcam_frames':   gradcam_frames,
            'original_video':   video_basename,
            'elapsed':          elapsed,
        }
        return render(request, 'predict.html', context)

    except FileNotFoundError as e:
        return render(request, 'error.html', {
            'error_title': 'Model Not Found',
            'error_message': str(e),
            'error_hint': 'Copy best_model_ffpp.pth into the django_app/models/ folder and restart the server.'
        })
    except Exception as e:
        return render(request, 'error.html', {
            'error_title': 'Prediction Failed',
            'error_message': str(e),
            'error_hint': 'Make sure your GPU has enough memory, or try a shorter video.'
        })


def about(request):
    """About page."""
    return render(request, 'about.html')
