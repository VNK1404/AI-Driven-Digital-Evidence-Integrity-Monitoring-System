"""
Rule-Based Metadata Analysis
===============================
Applies heuristic rules to extracted metadata to flag suspicious attributes.
Supports image, video, audio, and document file types.

Public API
----------
    metadata_rules(metadata, file_type="image") → list[str]
"""


# ---------------------------------------------------------------------------
# Editing-software keywords (case-insensitive)
# ---------------------------------------------------------------------------
_EDITING_SOFTWARE = [
    # Desktop NLEs & photo editors
    "photoshop", "gimp", "adobe", "lightroom", "capture one",
    "premiere", "after effects", "davinci", "resolve", "final cut",
    "avid", "vegas", "filmora", "handbrake", "ffmpeg", "ffprobe",
    "audacity", "sox", "adobe audition", "logic pro", "ableton",
    "imagemagick", "paint.net", "affinity", "corel", "canva",
    "snapseed", "vsco", "pixlr",
    # Mobile / consumer video editors
    "capcut", "inshot", "kinemaster", "powerdirector", "vn editor",
    "splice", "quik", "vivavideo", "videoshow", "magisto",
    "wevideo", "clips", "imovie", "luma fusion", "lumafusion",
    # NOTE: Generic codec library names (x264, x265, libx264, lavc, lavf, vlc)
    # are intentionally excluded — they appear inside standard H.264/H.265 codec
    # format reference URLs embedded by pymediainfo in ALL videos, including
    # authentic camera recordings shared via WhatsApp, Telegram, etc.
    # Only human-operated editing tool names are listed above.
]


# ---------------------------------------------------------------------------
# Per-type rule functions
# ---------------------------------------------------------------------------

def _common_rules(metadata: dict) -> list:
    """Rules that apply to ALL file types."""
    flags = []

    # Empty or near-empty metadata
    real_keys = [k for k in metadata if k not in ("extraction_error", "_extractor", "note")]
    if len(real_keys) <= 1:
        flags.append("empty_metadata")

    # Extraction error present
    if "extraction_error" in metadata:
        flags.append("extraction_failed")

    # Editing software detection
    meta_str = str(metadata).lower()
    for sw in _EDITING_SOFTWARE:
        if sw in meta_str:
            flags.append("editing_software_detected")
            break

    return flags


def _image_rules(metadata: dict) -> list:
    flags = []
    meta_str = str(metadata)

    if "Model" not in meta_str:
        flags.append("camera_model_missing")

    if "Make" not in meta_str:
        flags.append("camera_make_missing")

    # INFO — GPS coordinates are normal in authentic camera photos.
    # No score penalty is applied (see FLAG_SEVERITY in scoring.py).
    if "GPSInfo" in meta_str:
        flags.append("gps_data_present")

    # INFO — A software tag exists in many authentic files (e.g., camera firmware).
    # Only penalized if the software is a known *editing* tool (editing_software_detected).
    if "Software" in meta_str:
        flags.append("software_tag_present")

    return flags


def _video_rules(metadata: dict) -> list:
    flags = []
    meta_str = str(metadata).lower()

    if "encoder" not in meta_str and "codec" not in meta_str and "video_codec" not in meta_str:
        flags.append("encoder_info_missing")

    if "duration" not in meta_str:
        flags.append("duration_info_missing")

    if "creation" not in meta_str and "date" not in meta_str and "create" not in meta_str:
        flags.append("creation_date_missing")

    # Authentic camera recordings always embed the recording device/software.
    # We check broadly across standard fields AND vendor-specific atoms
    # (Android: comandroid*, Xiaomi/Samsung custom tags; iOS: quicktime*).
    # Only flag if TRULY zero device/encoder information is found.
    writing_keys = [
        # Standard fields
        "writing_application", "writing_library", "encoded_by",
        "handler_name", "handler", "encoder", "software",
        # Android / Xiaomi / Samsung atoms (appear as keys in full dump)
        "comandroid", "comxiaomi", "comsamsung", "manufacturer",
        # Apple / iOS QuickTime atoms
        "quicktime", "apple", "com_apple",
        # Generic device identifiers
        "make", "model", "device",
    ]
    if not any(k in meta_str for k in writing_keys):
        flags.append("writing_application_missing")

    # No tool could extract anything
    if metadata.get("_extractor") == "none":
        flags.append("no_extraction_tool_available")

    return flags


def _audio_rules(metadata: dict) -> list:
    flags = []

    sample_rate = metadata.get("sample_rate", 0)
    if isinstance(sample_rate, (int, float)):
        if 0 < sample_rate < 8000:
            flags.append("low_sample_rate")
        if sample_rate > 96000:
            flags.append("unusually_high_sample_rate")

    channels = metadata.get("channels", 0)
    if isinstance(channels, (int, float)) and channels == 0:
        flags.append("channel_info_missing")

    duration = metadata.get("duration", 0)
    if isinstance(duration, (int, float)) and duration <= 0:
        flags.append("zero_or_negative_duration")

    return flags


def _document_rules(metadata: dict) -> list:
    flags = []
    meta_str = str(metadata).lower()

    if "author" not in meta_str:
        flags.append("author_missing")

    if "producer" not in meta_str:
        flags.append("producer_missing")

    if "creator" not in meta_str:
        flags.append("creator_missing")

    if "creation_date" not in meta_str and "creationdate" not in meta_str:
        flags.append("creation_date_missing")

    return flags


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
_RULE_DISPATCH = {
    "image": _image_rules,
    "video": _video_rules,
    "audio": _audio_rules,
    "document": _document_rules,
}


def metadata_rules(metadata: dict, file_type: str = "image") -> list:
    """Apply rule-based checks to *metadata*.

    Flags are returned as plain strings and scored by ``compute_metadata_score``
    using a severity tier (HIGH / MEDIUM / INFO).  Not every flag is equally
    suspicious — see ``FLAG_SEVERITY`` in ``analysis/scoring.py`` for the
    point deductions associated with each flag.

    Parameters
    ----------
    metadata : dict
        Raw metadata dictionary from an extractor.
    file_type : str
        One of ``image``, ``video``, ``audio``, ``document``.

    Returns
    -------
    list[str]
        List of flag strings describing potential anomalies / informational notes.
    """
    flags = _common_rules(metadata)

    type_fn = _RULE_DISPATCH.get(file_type, _image_rules)
    flags.extend(type_fn(metadata))

    # De-duplicate while preserving order
    seen = set()
    unique = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            unique.append(f)

    return unique
