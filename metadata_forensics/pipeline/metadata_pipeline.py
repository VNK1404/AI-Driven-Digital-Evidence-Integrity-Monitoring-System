"""
Unified Metadata Forensic Pipeline
=====================================
Orchestrates the full forensic analysis:

    extract → features → rules → anomaly → score

Public API
----------
    run_metadata_pipeline(file_path, model=None, file_type=None) → dict
"""

from extraction import extract_metadata, detect_file_type
from extraction.image_extractor import extract_image_metadata
from extraction.video_extractor import extract_video_metadata
from extraction.audio_extractor import extract_audio_metadata
from extraction.document_extractor import extract_document_metadata
from analysis.features import metadata_to_features, features_to_vector
from analysis.rules import metadata_rules
from analysis.timeline import analyze_timeline
from analysis.scoring import compute_metadata_score
from analysis.anomaly_model import detect_anomaly


# Direct extractor dispatch (for when file_type is supplied explicitly)
_EXTRACTORS = {
    "image": extract_image_metadata,
    "video": extract_video_metadata,
    "audio": extract_audio_metadata,
    "document": extract_document_metadata,
}


def run_metadata_pipeline(file_path, model=None, file_type=None):
    """Execute the full metadata forensic pipeline on a single file.

    Parameters
    ----------
    file_path : str
        Absolute path to the evidence file.
    model : IsolationForest | None
        Optional pre-trained anomaly model from ``train_anomaly_model()``.
        If ``None``, uses zero-training reference-profile detection.
    file_type : str | None
        One of ``image``, ``video``, ``audio``, ``document``.
        If ``None``, auto-detected from file extension.

    Returns
    -------
    dict
        {
            "metadata_score": int,       # 0–100 integrity score
            "flags": list[str],          # rule-based flags
            "anomaly": bool,             # AI anomaly verdict
            "metadata": dict,            # raw extracted metadata
            "features": dict,            # engineered features
            "timeline_flags": list[str], # filesystem timeline flags
        }
    """
    # ----- Step 1: Determine file type ---------------------------------
    if file_type is None:
        file_type = detect_file_type(str(file_path))

    # ----- Step 2: Extract metadata ------------------------------------
    if file_type in _EXTRACTORS:
        try:
            metadata = _EXTRACTORS[file_type](str(file_path))
        except Exception as exc:
            metadata = {"extraction_error": str(exc)}
    else:
        try:
            result = extract_metadata(str(file_path))
            file_type = result["file_type"]
            metadata = result["metadata"]
        except Exception as exc:
            metadata = {"extraction_error": str(exc)}

    # Ensure metadata is always a dict
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": str(metadata)}

    # ----- Step 3: Feature engineering ---------------------------------
    try:
        features_dict = metadata_to_features(metadata, file_type)
        features_vec = features_to_vector(features_dict)
    except Exception:
        features_dict = {}
        features_vec = [0] * 7

    # ----- Step 4: Rule-based checks -----------------------------------
    try:
        flags = metadata_rules(metadata, file_type=file_type)
    except Exception:
        flags = ["rule_check_error"]

    # ----- Step 5: Timeline analysis -----------------------------------
    try:
        timeline_flags = analyze_timeline(str(file_path))
    except Exception:
        timeline_flags = []

    # ----- Step 6: AI anomaly detection --------------------------------
    try:
        anomaly = detect_anomaly(model, features_vec, file_type=file_type)
    except Exception:
        anomaly = False

    # ----- Step 7: Compute score ---------------------------------------
    try:
        score = compute_metadata_score(flags, anomaly)
    except Exception:
        score = -1

    return {
        "metadata_score": score,
        "flags": flags,
        "anomaly": anomaly,
        "metadata": metadata,
        "features": features_dict,
        "timeline_flags": timeline_flags,
        # Legacy aliases for backward compat with run_full_metadata_forensics.py
        "score": score,
    }
