"""
Unified Metadata Extraction Layer
===================================
Provides ``extract_metadata(file_path)`` which automatically detects the
file type and routes to the appropriate extractor.
"""

import os
from extraction.image_extractor import extract_image_metadata
from extraction.video_extractor import extract_video_metadata
from extraction.audio_extractor import extract_audio_metadata
from extraction.document_extractor import extract_document_metadata


# Extension → file type mapping
_EXTENSION_MAP = {
    # Images
    ".jpg": "image", ".jpeg": "image", ".png": "image",
    ".bmp": "image", ".tif": "image", ".tiff": "image",
    ".gif": "image", ".webp": "image",
    # Videos
    ".mp4": "video", ".avi": "video", ".mov": "video",
    ".mkv": "video", ".wmv": "video", ".flv": "video",
    ".webm": "video",
    # Audio
    ".wav": "audio", ".mp3": "audio", ".flac": "audio",
    ".aac": "audio", ".ogg": "audio", ".wma": "audio",
    ".m4a": "audio",
    # Documents
    ".pdf": "document",
}

# Extractor dispatch
_EXTRACTORS = {
    "image": extract_image_metadata,
    "video": extract_video_metadata,
    "audio": extract_audio_metadata,
    "document": extract_document_metadata,
}


def detect_file_type(file_path: str) -> str:
    """Detect file type from extension. Returns 'image', 'video', 'audio',
    'document', or 'unknown'."""
    ext = os.path.splitext(file_path)[1].lower()
    return _EXTENSION_MAP.get(ext, "unknown")


def extract_metadata(file_path: str) -> dict:
    """Unified metadata extraction entry point.

    Automatically detects the file type and calls the appropriate extractor.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to the evidence file.

    Returns
    -------
    dict
        ``{"file_type": str, "metadata": dict}``
        If the file type is unsupported the metadata dict will contain
        an ``extraction_error`` key.
    """
    file_type = detect_file_type(file_path)

    if file_type == "unknown":
        return {
            "file_type": "unknown",
            "metadata": {"extraction_error": f"Unsupported file extension: {os.path.splitext(file_path)[1]}"},
        }

    extractor = _EXTRACTORS[file_type]

    try:
        metadata = extractor(str(file_path))
    except Exception as exc:
        metadata = {"extraction_error": str(exc)}

    # Ensure metadata is always a dict
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": str(metadata)}

    return {
        "file_type": file_type,
        "metadata": metadata,
    }
