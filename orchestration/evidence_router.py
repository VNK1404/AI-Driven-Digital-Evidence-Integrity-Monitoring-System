"""
orchestration/evidence_router.py
----------------------------------
File-type detection and module routing rules.

Given a file path, this module determines:
  1. The broad file category: 'image', 'video', 'document', or 'unsupported'
  2. Which forensic modules should run on it

This is the SINGLE source of truth for routing.  The orchestrator and the
API both use this module — never duplicate routing logic elsewhere.

Routing Rules (from INTEGRATION_AUDIT.md — Part L)
---------------------------------------------------
IMAGE  → blockchain, metadata, image_forgery, [fake_news if applicable]
VIDEO  → blockchain, metadata, deepfake
DOCUMENT → blockchain, metadata, [fake_news if applicable]
UNSUPPORTED → reject

Audio is explicitly NOT supported.
"""

from pathlib import Path
from dataclasses import dataclass, field

from config.settings import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    DOCUMENT_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
)


# ── Routing result ─────────────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    """
    Describes which modules will run for a given evidence file.

    Attributes
    ----------
    file_path : Path
        Resolved absolute path to the evidence file.
    file_type : str
        'image' | 'video' | 'document' | 'unsupported'
    file_extension : str
        Lower-case extension with leading dot (e.g. '.jpg')
    is_supported : bool
        False when the extension is not in any supported category.
    run_blockchain : bool
        Always True for supported files.
    run_metadata : bool
        Always True for supported files.
    run_image_forgery : bool
        True only for image files.
    run_deepfake : bool
        True only for video files.
    run_fake_news : bool
        True for image and document files (news content possible).
        The fake_news adapter then decides if OCR yields usable text.
    reject_reason : str
        Set when is_supported is False.
    """
    file_path: Path
    file_type: str
    file_extension: str
    is_supported: bool
    run_blockchain: bool = False
    run_metadata: bool = False
    run_image_forgery: bool = False
    run_deepfake: bool = False
    run_fake_news: bool = False
    reject_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "file_path":        str(self.file_path),
            "file_type":        self.file_type,
            "file_extension":   self.file_extension,
            "is_supported":     self.is_supported,
            "run_blockchain":   self.run_blockchain,
            "run_metadata":     self.run_metadata,
            "run_image_forgery": self.run_image_forgery,
            "run_deepfake":     self.run_deepfake,
            "run_fake_news":    self.run_fake_news,
            "reject_reason":    self.reject_reason,
        }


# ── Audio extensions (explicitly rejected) ─────────────────────────────────────
_AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".aac", ".flac", ".ogg", ".m4a", ".wma", ".opus",
}


# ── Public API ─────────────────────────────────────────────────────────────────

def route_evidence(file_path: str) -> RoutingDecision:
    """
    Determine the file category and which modules to run.

    Parameters
    ----------
    file_path : str
        Path to the evidence file (may be relative or absolute).

    Returns
    -------
    RoutingDecision
        Always returns a decision object — never raises.
        Check `is_supported` before proceeding.
    """
    path = Path(file_path).resolve()
    ext = path.suffix.lower()

    # ── Audio: explicitly reject before any other check ───────────────────────
    if ext in _AUDIO_EXTENSIONS:
        return RoutingDecision(
            file_path=path,
            file_type="audio",
            file_extension=ext,
            is_supported=False,
            reject_reason=(
                f"Audio files are not supported by this system (extension: '{ext}'). "
                "Audio forensics has been removed from the project scope."
            ),
        )

    # ── Image ─────────────────────────────────────────────────────────────────
    if ext in IMAGE_EXTENSIONS:
        return RoutingDecision(
            file_path=path,
            file_type="image",
            file_extension=ext,
            is_supported=True,
            run_blockchain=True,
            run_metadata=True,
            run_image_forgery=True,
            run_deepfake=False,
            run_fake_news=True,    # conditional — adapter checks OCR quality
        )

    # ── Video ─────────────────────────────────────────────────────────────────
    if ext in VIDEO_EXTENSIONS:
        return RoutingDecision(
            file_path=path,
            file_type="video",
            file_extension=ext,
            is_supported=True,
            run_blockchain=True,
            run_metadata=True,
            run_image_forgery=False,
            run_deepfake=True,
            run_fake_news=False,
        )

    # ── Document ──────────────────────────────────────────────────────────────
    if ext in DOCUMENT_EXTENSIONS:
        return RoutingDecision(
            file_path=path,
            file_type="document",
            file_extension=ext,
            is_supported=True,
            run_blockchain=True,
            run_metadata=True,
            run_image_forgery=False,
            run_deepfake=False,
            run_fake_news=True,    # conditional — adapter checks OCR quality
        )

    # ── Unsupported ───────────────────────────────────────────────────────────
    return RoutingDecision(
        file_path=path,
        file_type="unsupported",
        file_extension=ext,
        is_supported=False,
        reject_reason=(
            f"Unsupported file type: '{ext}'. "
            f"Supported: images {sorted(IMAGE_EXTENSIONS)}, "
            f"videos {sorted(VIDEO_EXTENSIONS)}, "
            f"documents {sorted(DOCUMENT_EXTENSIONS)}."
        ),
    )


def describe_routing(decision: RoutingDecision) -> str:
    """Human-readable summary of routing decision (for logging)."""
    if not decision.is_supported:
        return f"REJECTED: {decision.reject_reason}"

    modules = []
    if decision.run_blockchain:   modules.append("blockchain")
    if decision.run_metadata:     modules.append("metadata")
    if decision.run_image_forgery: modules.append("image_forgery")
    if decision.run_deepfake:     modules.append("deepfake")
    if decision.run_fake_news:    modules.append("fake_news[conditional]")

    return (
        f"{decision.file_type.upper()} [{decision.file_extension}] -> "
        f"{' -> '.join(modules)}"
    )
