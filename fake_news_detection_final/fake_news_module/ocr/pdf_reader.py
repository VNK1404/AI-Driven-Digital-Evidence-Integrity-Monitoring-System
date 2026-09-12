"""
ocr/pdf_reader.py — PDF text extraction using pdfplumber
AI-Driven Digital Evidence Integrity Monitoring System
"""

import logging
from pathlib import Path
from typing import Optional

try:
    import pdfplumber
except ImportError as exc:
    raise ImportError("pdfplumber is required: pip install pdfplumber") from exc

logger = logging.getLogger(__name__)


def _extract_page_text(page) -> str:
    """
    Extract text from a single pdfplumber page object.
    Returns empty string if extraction yields nothing.
    """
    try:
        raw = page.extract_text(x_tolerance=2, y_tolerance=2)
        return raw.strip() if raw else ""
    except Exception as exc:                          # noqa: BLE001
        logger.warning("Failed to extract text from page: %s", exc)
        return ""


def extract_text_from_pdf(file_path: str, max_pages: Optional[int] = None) -> str:
    """
    Extract full text from a PDF file.

    Args:
        file_path:  Absolute path to the PDF.
        max_pages:  Optional limit on number of pages to read (None = all).

    Returns:
        Concatenated text from all pages.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If the file is not a valid PDF.
        RuntimeError:      If pdfplumber encounters an unrecoverable error.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    logger.info("Opening PDF: %s", path.name)

    try:
        with pdfplumber.open(str(path)) as pdf:
            pages = pdf.pages
            if max_pages is not None:
                pages = pages[:max_pages]

            page_texts = []
            for i, page in enumerate(pages, start=1):
                text = _extract_page_text(page)
                if text:
                    page_texts.append(text)
                    logger.debug("Page %d: %d characters extracted.", i, len(text))
                else:
                    logger.debug("Page %d: no text found (possibly image-based).", i)

    except pdfplumber.pdfminer.pdfparser.PDFSyntaxError as exc:
        raise ValueError(f"Invalid or corrupted PDF file: {file_path}") from exc
    except Exception as exc:                          # noqa: BLE001
        raise RuntimeError(f"PDF extraction failed: {exc}") from exc

    full_text = "\n\n".join(page_texts).strip()
    logger.info(
        "PDF extraction complete. %d pages processed, %d total characters.",
        len(page_texts),
        len(full_text),
    )
    return full_text
