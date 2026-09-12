"""
Video Metadata Extractor
=========================
Extracts metadata from MP4, AVI, MOV, MKV files.

Extraction order
----------------
1. ExifTool  (subprocess, most detailed)
2. ffprobe   (subprocess, FFmpeg)
3. pymediainfo (pure-Python, no external binary needed) ← NEW fallback

Always returns a dict, never a raw string.
"""

import subprocess
import json
import shutil


def _parse_exiftool_output(raw: str) -> dict:
    """Parse ExifTool's default colon-delimited output into a dict."""
    metadata = {}
    for line in raw.strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if key and value:
                metadata[key] = value
    return metadata


def _extract_via_exiftool(path: str) -> dict:
    """Extract metadata using ExifTool."""
    exiftool_cmd = shutil.which("exiftool")
    if not exiftool_cmd:
        return {}

    try:
        result = subprocess.run(
            [exiftool_cmd, "-s", "-s", "-s", "-G", path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return _parse_exiftool_output(result.stdout)
    except Exception:
        pass
    return {}


def _extract_via_ffprobe(path: str) -> dict:
    """Extract metadata using ffprobe (FFmpeg) as fallback."""
    ffprobe_cmd = shutil.which("ffprobe")
    if not ffprobe_cmd:
        return {}

    try:
        result = subprocess.run(
            [
                ffprobe_cmd, "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            metadata = {}

            # Format-level metadata
            fmt = data.get("format", {})
            if fmt.get("format_name"):
                metadata["format_name"] = fmt["format_name"]
            if fmt.get("duration"):
                metadata["duration"] = fmt["duration"]
            if fmt.get("bit_rate"):
                metadata["bit_rate"] = fmt["bit_rate"]

            # Tags
            for key, val in fmt.get("tags", {}).items():
                metadata[f"tag_{key}"] = val

            # First video stream
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    metadata["video_codec"] = stream.get("codec_name", "")
                    metadata["video_width"] = stream.get("width", 0)
                    metadata["video_height"] = stream.get("height", 0)
                    metadata["video_fps"] = stream.get("r_frame_rate", "")
                    break

            # First audio stream
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "audio":
                    metadata["audio_codec"] = stream.get("codec_name", "")
                    metadata["audio_sample_rate"] = stream.get("sample_rate", "")
                    metadata["audio_channels"] = stream.get("channels", 0)
                    break

            return metadata
    except Exception:
        pass
    return {}


def _extract_via_pymediainfo(path: str) -> dict:
    """Extract metadata using pymediainfo (pure-Python, no binary needed).

    Dumps ALL available non-null fields from every track so no editing
    software fingerprint (writing_application, encoder, handler, etc.) is
    ever silently dropped.
    """
    try:
        from pymediainfo import MediaInfo
        media = MediaInfo.parse(path)
        metadata = {}

        for track in media.tracks:
            t = track.track_type  # 'General', 'Video', 'Audio', ...
            prefix = "" if t == "General" else t.lower() + "_"

            # Dump every non-null field from this track
            for key, val in track.to_data().items():
                if val is None or val == "" or key == "track_type":
                    continue
                # Normalize key: lowercase, spaces→underscores
                norm_key = f"{prefix}{key.lower().replace(' ', '_')}"
                metadata[norm_key] = val

            # ── Structured aliases used by rules / anomaly detection ──────
            if t == "General":
                if track.duration and "duration" not in metadata:
                    metadata["duration"] = round(float(track.duration) / 1000, 3)
                elif "duration" in metadata:
                    try:
                        metadata["duration"] = round(float(metadata["duration"]) / 1000, 3)
                    except (ValueError, TypeError):
                        pass

            elif t == "Video":
                codec = track.codec_id or track.format or ""
                if codec:
                    metadata["video_codec"] = codec
                if track.width:
                    metadata["video_width"] = track.width
                if track.height:
                    metadata["video_height"] = track.height
                if track.frame_rate:
                    metadata["video_fps"] = track.frame_rate

            elif t == "Audio":
                codec = track.codec_id or track.format or ""
                if codec:
                    metadata["audio_codec"] = codec
                if track.sampling_rate:
                    metadata["audio_sample_rate"] = track.sampling_rate
                if track.channel_s:
                    metadata["audio_channels"] = track.channel_s

        return metadata if metadata else {}

    except ImportError:
        return {}
    except Exception:
        return {}


def extract_video_metadata(path: str) -> dict:
    """Extract metadata from a video file.

    Tries ExifTool first, then ffprobe, then pymediainfo.
    Returns a dict of metadata key-value pairs.
    Returns a minimal error dict if all methods fail.
    """
    # 1. Try ExifTool
    metadata = _extract_via_exiftool(path)
    if metadata:
        metadata["_extractor"] = "exiftool"
        return metadata

    # 2. Fallback to ffprobe
    metadata = _extract_via_ffprobe(path)
    if metadata:
        metadata["_extractor"] = "ffprobe"
        return metadata

    # 3. Fallback to pymediainfo (pure-Python, no binary needed)
    metadata = _extract_via_pymediainfo(path)
    if metadata:
        metadata["_extractor"] = "pymediainfo"
        return metadata

    return {"_extractor": "none", "note": "Neither exiftool, ffprobe, nor pymediainfo could extract metadata"}
