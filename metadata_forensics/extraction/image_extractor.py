"""
Image Metadata Extractor
=========================
Extracts EXIF and image-level metadata from JPEG, PNG, TIFF, BMP files
using Pillow. Handles missing EXIF, corrupted files, and binary values.
"""

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


def _safe_value(val):
    """Convert EXIF value to a JSON-serialisable type."""
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8", errors="replace")
        except Exception:
            return f"<binary {len(val)} bytes>"
    if isinstance(val, tuple) and len(val) == 2 and all(isinstance(v, int) for v in val):
        # IFDRational — convert to float
        return val[0] / val[1] if val[1] != 0 else 0.0
    if isinstance(val, dict):
        return {str(k): _safe_value(v) for k, v in val.items()}
    return val


def _parse_gps(gps_info):
    """Parse GPS IFD into readable dict."""
    parsed = {}
    for key, val in gps_info.items():
        tag_name = GPSTAGS.get(key, key)
        parsed[str(tag_name)] = _safe_value(val)
    return parsed


def extract_image_metadata(path: str) -> dict:
    """Extract metadata from an image file.

    Supports JPEG (EXIF), PNG (text chunks), TIFF, and BMP.
    Returns a flat dictionary of metadata key-value pairs.
    Returns an empty dict (not an error) when metadata is absent.
    """
    metadata = {}

    try:
        img = Image.open(path)
    except Exception as exc:
        return {"extraction_error": f"Cannot open image: {exc}"}

    # --- Basic image properties (always available) ---
    metadata["image_format"] = img.format or "unknown"
    metadata["image_mode"] = img.mode
    metadata["image_width"] = img.size[0]
    metadata["image_height"] = img.size[1]

    # --- EXIF data (JPEG, TIFF) ---
    try:
        exif_data = img._getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                if tag_name == "GPSInfo" and isinstance(value, dict):
                    metadata["GPSInfo"] = _parse_gps(value)
                elif tag_name == "MakerNote":
                    metadata["MakerNote"] = "<present>"
                else:
                    metadata[tag_name] = _safe_value(value)
    except Exception:
        pass  # No EXIF — not an error

    # --- PNG text chunks ---
    if img.format == "PNG":
        try:
            png_info = img.info
            if png_info:
                for key, val in png_info.items():
                    if isinstance(val, (str, int, float)):
                        metadata[f"png_{key}"] = val
        except Exception:
            pass

    return metadata
