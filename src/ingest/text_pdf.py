"""Purpose: decode raw text/PDF bytes into strings suitable for embedding.
Why extend: plug in richer PDF parsing or support new text-like formats without touching the main pipeline.
How extend: implement additional functions (e.g. `extract_markdown_sections`) and call them from the pipeline when handling new extensions.
"""
from __future__ import annotations

import io
import warnings
from typing import List

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from src.ingest.constants import PDF_TEXT_CHAR_LIMIT


def extract_pdf_text(blob: bytes, max_pages: int) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            reader = PdfReader(io.BytesIO(blob))
        except (PdfReadError, Exception):
            return ""
    parts: List[str] = []
    for i, page in enumerate(reader.pages):
        if i >= max_pages:
            break
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if txt:
            parts.append(txt)
    text = "\n".join(parts)
    return text[:PDF_TEXT_CHAR_LIMIT] if len(text) > PDF_TEXT_CHAR_LIMIT else text


def decode_text_bytes(blob: bytes) -> str:
    try:
        return blob.decode("utf-8", errors="ignore")
    except Exception:
        return ""
