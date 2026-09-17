"""Lightweight PDF text extraction for RAG / Deep Research.

Default path for MorphAgent UI / demo: extract embedded text with PyMuPDF
(or pypdf), then let the LLM summarize. This avoids the multi-minute
PaddleX layout_parsing OCR stack on CPU.

Optional: set ``RAG_PDF_BACKEND=paddlex`` (or ``PDF_TEXT_BACKEND=paddlex``)
to force the heavy layout-aware PaddleX path when a CUDA GPU is available.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Soft cap so a single long paper does not blow the LLM context before batching.
_DEFAULT_MAX_CHARS = int(os.getenv("PDF_TEXT_MAX_CHARS", "80000"))


def _backend_name() -> str:
    return (
        os.getenv("RAG_PDF_BACKEND")
        or os.getenv("PDF_TEXT_BACKEND")
        or "lite"
    ).strip().lower()


def extract_text_from_pdf_lite(
    pdf_path: Path,
    *,
    max_chars: Optional[int] = None,
) -> str:
    """Extract plain text from a PDF without layout OCR.

    Tries PyMuPDF first, then pypdf. Returns an ``[ERROR] ...`` string on failure
    (callers treat empty/error strings as skippable).
    """
    path = Path(pdf_path)
    if not path.is_file():
        return f"[ERROR] PDF not found: {path}"

    limit = _DEFAULT_MAX_CHARS if max_chars is None else int(max_chars)
    text = ""
    method = ""

    try:
        import fitz  # PyMuPDF

        doc = fitz.open(path)
        parts: list[str] = []
        for page in doc:
            parts.append(page.get_text("text") or "")
        doc.close()
        text = "\n".join(parts).strip()
        method = "pymupdf"
    except Exception as pymupdf_err:
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            parts = []
            for page in reader.pages:
                parts.append(page.extract_text() or "")
            text = "\n".join(parts).strip()
            method = "pypdf"
        except Exception as pypdf_err:
            return (
                f"[ERROR] Lightweight PDF extraction failed for {path.name}: "
                f"pymupdf={pymupdf_err!r}; pypdf={pypdf_err!r}. "
                "Install pymupdf (recommended) or pypdf."
            )

    if not text:
        return (
            f"[ERROR] No extractable text in {path.name} "
            f"(scanned/image-only PDF; lite backend={method})."
        )

    if limit > 0 and len(text) > limit:
        omitted = len(text) - limit
        text = (
            text[:limit]
            + f"\n\n[... truncated {omitted} characters for LLM context "
            f"(backend={method}, max_chars={limit}) ...]"
        )
    return text


def extract_text_from_pdf(
    pdf_path: Path,
    *,
    device: str = "cpu",
    max_chars: Optional[int] = None,
) -> str:
    """Extract PDF text using the configured backend (lite by default)."""

    backend = _backend_name()
    if backend in {"paddlex", "paddle", "layout"}:
        try:
            from .paddlex_loader import PaddleXPDFLoader
        except ImportError:
            print(
                "  [PDF] RAG_PDF_BACKEND=paddlex requested but PaddleX is not installed; "
                "falling back to lite extraction"
            )
            return extract_text_from_pdf_lite(pdf_path, max_chars=max_chars)
        try:
            loader = PaddleXPDFLoader(str(pdf_path), device=device or "cpu")
            documents = loader.load()
            text = "\n\n".join(doc.page_content for doc in documents if doc.page_content).strip()
            if not text:
                return f"[ERROR] PaddleX returned no text for {Path(pdf_path).name}"
            limit = _DEFAULT_MAX_CHARS if max_chars is None else int(max_chars)
            if limit > 0 and len(text) > limit:
                omitted = len(text) - limit
                text = (
                    text[:limit]
                    + f"\n\n[... truncated {omitted} characters for LLM context "
                    f"(backend=paddlex, max_chars={limit}) ...]"
                )
            return text
        except Exception as e:
            print(f"  [PDF] PaddleX failed ({e}); falling back to lite extraction")
            return extract_text_from_pdf_lite(pdf_path, max_chars=max_chars)

    return extract_text_from_pdf_lite(pdf_path, max_chars=max_chars)
