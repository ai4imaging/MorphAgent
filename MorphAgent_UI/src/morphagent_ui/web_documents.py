"""Bounded, local reference extraction. Uploaded documents are data, not code."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from xml.etree import ElementTree

MAX_DOCUMENT_BYTES = 20 * 1024 * 1024
MAX_TEXT_CHARACTERS = 200_000


def extract_document(name: str, content: bytes) -> str:
    if len(content) > MAX_DOCUMENT_BYTES:
        raise ValueError('Reference files must be smaller than 20 MB.')
    suffix = Path(name).suffix.lower()
    try:
        if suffix in {'.txt', '.md'}:
            text = content.decode('utf-8-sig')
        elif suffix == '.docx':
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                info = archive.getinfo('word/document.xml')
                if info.file_size > MAX_DOCUMENT_BYTES:
                    raise ValueError('The Word document is too large after decompression.')
                xml = archive.read(info)
                if b'<!DOCTYPE' in xml or b'<!ENTITY' in xml:
                    raise ValueError('XML entities are not supported.')
                root = ElementTree.fromstring(xml)
                ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                text = '\n'.join(''.join(p.itertext()) for p in root.findall('.//w:p', ns))
        elif suffix == '.pdf':
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise ValueError('PDF support needs pypdf. Install the updated UI requirements.') from exc
            reader = PdfReader(io.BytesIO(content))
            if len(reader.pages) > 300:
                raise ValueError('Please provide a PDF with at most 300 pages.')
            text = '\n'.join(page.extract_text() or '' for page in reader.pages)
        else:
            raise ValueError('Supported references: PDF, Word (.docx), UTF-8 TXT, and Markdown. Convert .doc to .docx first.')
    except (UnicodeError, zipfile.BadZipFile, KeyError, ElementTree.ParseError) as exc:
        raise ValueError('This document could not be read. Use UTF-8 text or a valid PDF/DOCX.') from exc
    text = text.replace('\x00', '').strip()
    if not text:
        raise ValueError('No readable text found. Scanned PDFs need OCR before uploading.')
    if len(text) > MAX_TEXT_CHARACTERS:
        raise ValueError('Reference text exceeds 200,000 characters. Please select relevant sections.')
    return text
