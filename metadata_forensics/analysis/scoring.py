"""
Metadata Scoring System
=========================
Computes an integrity score from rule-based flags and anomaly detection.

Public API
----------
    compute_metadata_score(flags, anomaly) → int  (0–100)

Flag Severity Tiers
-------------------
Each known flag is assigned a severity tier that controls how many points
it deducts from the base score of 100:

    HIGH   → −25 pts  (strong evidence of tampering / critical data missing)
    MEDIUM → −10 pts  (suspicious, but common in authentic stripped-EXIF files)
    INFO   →   0 pts  (notable, but normal in real-world files; no penalty)

Unknown / unlisted flags default to MEDIUM (−10 pts).

Anomaly Penalty
---------------
If the AI anomaly detector fires, −20 pts are deducted.  This is intentionally
moderate because the detector uses statistical reference profiles, not ground
truth — false positives are possible.
"""


# ---------------------------------------------------------------------------
# Severity → point deduction mapping
# ---------------------------------------------------------------------------

FLAG_SEVERITY = {
    # ── HIGH (−25) ──────────────────────────────────────────────────────────
    # Direct evidence of editing tools or complete metadata absence.
    "editing_software_detected":        "HIGH",
    "extraction_failed":                "HIGH",
    "empty_metadata":                   "HIGH",
    "no_extraction_tool_available":     "HIGH",

    # ── MEDIUM (−10) ────────────────────────────────────────────────────────
    # Absent fields that are *expected* in authentic files but can be
    # legitimately stripped (e.g., privacy-scrubbed or web-compressed files).
    "camera_model_missing":             "MEDIUM",
    "camera_make_missing":              "MEDIUM",
    "encoder_info_missing":             "MEDIUM",
    "duration_info_missing":            "MEDIUM",
    "creation_date_missing":            "MEDIUM",
    "writing_application_missing":      "MEDIUM",
    "author_missing":                   "MEDIUM",
    "producer_missing":                 "MEDIUM",
    "creator_missing":                  "MEDIUM",
    "low_sample_rate":                  "MEDIUM",
    "zero_or_negative_duration":        "MEDIUM",
    "channel_info_missing":             "MEDIUM",
    "unusually_high_sample_rate":       "MEDIUM",
    "rule_check_error":                 "MEDIUM",
    "file modified after creation":     "MEDIUM",

    # ── INFO (0) ─────────────────────────────────────────────────────────────
    # Present in many authentic files; worth noting but not penalizing.
    "gps_data_present":                 "INFO",
    "software_tag_present":             "INFO",
}

_SEVERITY_POINTS = {
    "HIGH":   25,
    "MEDIUM": 10,
    "INFO":    0,
}

_ANOMALY_PENALTY = 20


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_metadata_score(flags: list, anomaly: bool = False) -> int:
    """Compute a metadata integrity score.

    Logic
    -----
    - Start with ``score = 100``
    - For each flag in *flags*, subtract the points for its severity tier
      (HIGH=25, MEDIUM=10, INFO=0).  Unknown flags default to MEDIUM.
    - Subtract **20** if *anomaly* is ``True``
    - Clamp the result to ``[0, 100]``

    Parameters
    ----------
    flags : list[str]
        List of flag strings from ``metadata_rules()`` and/or timeline analysis.
    anomaly : bool
        Whether the AI anomaly detector flagged this sample.

    Returns
    -------
    int
        Integrity score between 0 (highly suspicious) and 100 (clean).
    """
    score = 100

    for flag in flags:
        severity = FLAG_SEVERITY.get(flag, "MEDIUM")
        score -= _SEVERITY_POINTS[severity]

    if anomaly:
        score -= _ANOMALY_PENALTY

    return max(0, min(100, score))


# ---------------------------------------------------------------------------
# Backward compatibility alias
# ---------------------------------------------------------------------------

def compute_score(flags: list) -> int:
    """Legacy wrapper — calls ``compute_metadata_score`` with anomaly=False."""
    return compute_metadata_score(flags, anomaly=False)
