"""
Metadata Feature Engineering
==============================
Converts raw metadata dictionaries into numeric feature vectors
for anomaly detection and machine learning.
"""


# ---------------------------------------------------------------------------
# Editing‑software keywords (case-insensitive matching)
# ---------------------------------------------------------------------------
_EDITING_SOFTWARE = [
    "photoshop", "gimp", "adobe", "lightroom", "capture one",
    "premiere", "after effects", "davinci", "resolve", "final cut",
    "avid", "vegas", "filmora", "handbrake", "ffmpeg", "ffprobe",
    "audacity", "sox", "adobe audition", "logic pro", "ableton",
    "imagemagick", "paint.net", "affinity", "corel", "canva",
    "snapseed", "vsco", "pixlr",
]


def metadata_to_features(metadata: dict, file_type: str = "image") -> dict:
    """Convert a metadata dict into numeric features for anomaly detection.

    Parameters
    ----------
    metadata : dict
        Raw metadata as returned by any extractor.
    file_type : str
        One of ``image``, ``video``, ``audio``, ``document``.

    Returns
    -------
    dict
        Feature dictionary with the following keys:

        - ``camera_present`` (1/0)
        - ``editing_software_present`` (1/0)
        - ``gps_present`` (1/0)
        - ``metadata_length`` (int — number of metadata keys)
        - ``device_info_present`` (1/0)
        - ``timestamp_present`` (1/0)
        - ``has_error`` (1/0)
    """
    meta_str = str(metadata).lower()
    keys_lower = [str(k).lower() for k in metadata.keys()]

    features = {}

    # 1. camera_present — camera Make or Model in metadata
    features["camera_present"] = int(
        any(k in meta_str for k in ["model", "make", "camera"])
        and "no camera" not in meta_str
    )

    # 2. editing_software_present
    features["editing_software_present"] = int(
        any(sw in meta_str for sw in _EDITING_SOFTWARE)
    )

    # 3. gps_present
    features["gps_present"] = int(
        any(k in meta_str for k in ["gps", "latitude", "longitude", "location"])
    )

    # 4. metadata_length — total key count, capped to prevent pymediainfo's
    #    verbose internal fields (other_*, count_*, etc.) from inflating z-scores
    features["metadata_length"] = min(len(metadata), 50)

    # 5. device_info_present — any device/hardware identifiers
    features["device_info_present"] = int(
        any(k in meta_str for k in [
            "make", "model", "device", "hardware", "manufacturer",
            "lens", "serial", "firmware",
        ])
    )

    # 6. timestamp_present — creation/modification dates
    features["timestamp_present"] = int(
        any(k in meta_str for k in [
            "date", "time", "created", "modified", "creation",
            "datetime", "timestamp",
        ])
    )

    # 7. has_error — extraction failed
    features["has_error"] = int(
        "extraction_error" in metadata or "error" in keys_lower
    )

    return features


def features_to_vector(features: dict) -> list:
    """Convert the feature dict into a flat numeric list (consistent ordering).

    The ordering is deterministic and matches the keys returned by
    ``metadata_to_features``.
    """
    _ORDER = [
        "camera_present",
        "editing_software_present",
        "gps_present",
        "metadata_length",
        "device_info_present",
        "timestamp_present",
        "has_error",
    ]
    return [features.get(k, 0) for k in _ORDER]
