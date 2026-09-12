"""
schemas/evidence.py
--------------------
Core evidence dataclass — the central object that flows through
every module in the integration layer.

Every uploaded evidence item receives one unique evidence_id.
This object is created by the orchestrator before any module is called.
"""

import hashlib
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _generate_evidence_id() -> str:
    """
    Generate a deterministic, 6-digit, zero-padded evidence ID
    using a random UUID suffix.

    Format: EV-XXXXXX  (e.g.  EV-000042)
    In practice the numeric suffix is random (not sequential) so IDs
    don't leak the total count of evidence items.
    """
    suffix = uuid.uuid4().int % 1_000_000
    return f"EV-{suffix:06d}"


def compute_sha256(file_path: str | Path) -> str:
    """
    Compute the SHA-256 hash of a file by streaming it in 64-KB chunks.
    Works for files of any size (images, videos, PDFs).

    Args:
        file_path: Path to the file.

    Returns:
        SHA-256 hex digest string.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    sha256 = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


@dataclass
class EvidenceObject:
    """
    Represents a single piece of forensic evidence submitted for analysis.

    Attributes
    ----------
    evidence_id : str
        Unique identifier, auto-generated as EV-XXXXXX.
    file_path : Path
        Absolute path to the evidence file on disk.
    file_name : str
        Original file name (basename).
    file_type : str
        Broad category: 'image', 'video', or 'document'.
    file_extension : str
        Lower-case extension including the leading dot (e.g. '.jpg').
    size_bytes : int
        File size in bytes.
    sha256 : str
        SHA-256 hex digest computed at creation time.
    submitted_by : str
        Name or ID of the submitting user/system.
    submitted_at : str
        ISO-8601 UTC timestamp of when this object was created.
    mime_type : str
        Best-guess MIME type derived from the extension.
    """

    file_path: Path
    file_type: str
    evidence_id: str = field(default_factory=_generate_evidence_id)
    submitted_by: str = "system"
    submitted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Computed on post-init
    file_name: str = field(init=False)
    file_extension: str = field(init=False)
    size_bytes: int = field(init=False)
    sha256: str = field(init=False)
    mime_type: str = field(init=False)

    _MIME_MAP: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.file_path = Path(self.file_path).resolve()
        self.file_name = self.file_path.name
        self.file_extension = self.file_path.suffix.lower()
        self.size_bytes = self.file_path.stat().st_size
        self.sha256 = compute_sha256(self.file_path)
        self.mime_type = self._detect_mime()

    def _detect_mime(self) -> str:
        _map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".bmp": "image/bmp",
            ".tif": "image/tiff", ".tiff": "image/tiff",
            ".webp": "image/webp",
            ".mp4": "video/mp4", ".avi": "video/x-msvideo",
            ".mov": "video/quicktime", ".mkv": "video/x-matroska",
            ".webm": "video/webm", ".wmv": "video/x-ms-wmv",
            ".flv": "video/x-flv",
            ".pdf": "application/pdf",
        }
        return _map.get(self.file_extension, "application/octet-stream")

    def to_dict(self) -> dict:
        """Serialise to a plain JSON-friendly dictionary."""
        return {
            "evidence_id":    self.evidence_id,
            "file_name":      self.file_name,
            "file_path":      str(self.file_path),
            "file_type":      self.file_type,
            "file_extension": self.file_extension,
            "size_bytes":     self.size_bytes,
            "sha256":         self.sha256,
            "mime_type":      self.mime_type,
            "submitted_by":   self.submitted_by,
            "submitted_at":   self.submitted_at,
        }
