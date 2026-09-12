"""
hashing.py
----------
Universal SHA-256 hasher that works for ANY file type:
  - Images  : jpg, png, tif, bmp
  - Videos  : mp4, avi, mkv, mov
  - Audio   : mp3, wav, aac
  - Documents: pdf, docx, txt, xlsx
"""

import hashlib
import os

# Supported file types grouped by category
SUPPORTED_TYPES = {
    "image":    {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"},
    "video":    {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"},
    "audio":    {".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a"},
    "document": {".pdf", ".docx", ".doc", ".txt", ".xlsx", ".xls", ".pptx"},
}

# Flat set of all supported extensions
ALL_SUPPORTED = {ext for exts in SUPPORTED_TYPES.values() for ext in exts}


def get_file_type(file_path: str) -> str:
    """
    Determine the category of a file based on its extension.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: One of 'image', 'video', 'audio', 'document', or 'unknown'.
    """
    ext = os.path.splitext(file_path)[1].lower()
    for category, extensions in SUPPORTED_TYPES.items():
        if ext in extensions:
            return category
    return "unknown"


def is_supported(file_path: str) -> bool:
    """
    Check if the file type is supported by the system.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if supported, False otherwise.
    """
    ext = os.path.splitext(file_path)[1].lower()
    return ext in ALL_SUPPORTED


def generate_hash(file_path: str) -> str:
    """
    Generate a SHA-256 hash for any file type.

    Reads file in 64KB binary chunks — handles large video
    files (several GB) without memory issues.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: SHA-256 hex digest string.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file type is not supported.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not is_supported(file_path):
        ext = os.path.splitext(file_path)[1]
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(sorted(ALL_SUPPORTED))}"
        )

    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def get_file_metadata(file_path: str) -> dict:
    """
    Extract basic metadata from a file.

    Args:
        file_path (str): Path to the file.

    Returns:
        dict: file_name, file_size (bytes), file_type (category), extension.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return {
        "file_name":  os.path.basename(file_path),
        "file_size":  os.path.getsize(file_path),
        "file_type":  get_file_type(file_path),
        "extension":  os.path.splitext(file_path)[1].lower(),
    }
