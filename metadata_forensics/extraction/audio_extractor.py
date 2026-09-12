"""
Audio Metadata Extractor
=========================
Extracts metadata from WAV, MP3, FLAC, AAC, OGG files using Mutagen.
Uses auto-detection instead of hardcoded format classes.
"""

import mutagen


def extract_audio_metadata(path: str) -> dict:
    """Extract metadata from an audio file.

    Supports WAV, MP3, FLAC, OGG, AAC, and other Mutagen-supported formats.
    Returns a dict with audio properties and tags.
    """
    metadata = {}

    try:
        audio = mutagen.File(path, easy=False)
    except Exception as exc:
        return {"extraction_error": f"Cannot open audio: {exc}"}

    if audio is None:
        return {"extraction_error": "Unsupported or unrecognisable audio format"}

    # --- Core audio properties ---
    if audio.info:
        info = audio.info
        metadata["duration"] = getattr(info, "length", 0.0)
        metadata["sample_rate"] = getattr(info, "sample_rate", 0)
        metadata["channels"] = getattr(info, "channels", 0)
        metadata["bitrate"] = getattr(info, "bitrate", 0)
        metadata["bits_per_sample"] = getattr(info, "bits_per_sample", 0)

    # --- Tags / ID3 / Vorbis comments ---
    if audio.tags:
        for key in audio.tags:
            try:
                val = audio.tags[key]
                # Mutagen returns list-like objects for some tag types
                if hasattr(val, "text"):
                    metadata[f"tag_{key}"] = str(val.text[0]) if val.text else ""
                elif isinstance(val, list):
                    metadata[f"tag_{key}"] = str(val[0]) if val else ""
                else:
                    metadata[f"tag_{key}"] = str(val)
            except Exception:
                metadata[f"tag_{key}"] = "<unreadable>"

    # --- Mutagen type info ---
    metadata["audio_type"] = type(audio).__name__

    return metadata
