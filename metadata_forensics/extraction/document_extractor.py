"""
Document Metadata Extractor
=============================
Extracts metadata from PDF files using PyPDF2.
Handles encrypted PDFs, missing metadata, and internal object types.
"""

import PyPDF2


def extract_document_metadata(path: str) -> dict:
    """Extract metadata from a PDF document.

    Returns a plain dict with author, creator, producer, dates, page count.
    Handles encrypted PDFs (tries empty password), missing metadata.
    """
    metadata = {}

    try:
        reader = PyPDF2.PdfReader(path)
    except Exception as exc:
        return {"extraction_error": f"Cannot open PDF: {exc}"}

    # Handle encrypted PDFs
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            return {"extraction_error": "PDF is encrypted and could not be decrypted"}

    # --- Page count ---
    try:
        metadata["page_count"] = len(reader.pages)
    except Exception:
        metadata["page_count"] = 0

    # --- Document info ---
    try:
        info = reader.metadata
        if info:
            # Standard PDF metadata fields
            field_map = {
                "/Author": "author",
                "/Creator": "creator",
                "/Producer": "producer",
                "/Title": "title",
                "/Subject": "subject",
                "/Keywords": "keywords",
                "/CreationDate": "creation_date",
                "/ModDate": "modification_date",
            }
            for pdf_key, our_key in field_map.items():
                val = info.get(pdf_key, None)
                if val:
                    metadata[our_key] = str(val)

            # Capture any extra custom metadata
            for key in info:
                clean_key = str(key).lstrip("/")
                if clean_key not in metadata and clean_key.lower() not in [
                    v for v in field_map.values()
                ]:
                    val = info[key]
                    if val:
                        metadata[clean_key] = str(val)
    except Exception:
        pass

    return metadata
