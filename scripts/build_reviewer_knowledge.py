#!/usr/bin/env python3
"""Build the deterministic Ask MorphAgent reviewer knowledge bundle.

The generated JSON contains textual submission material and selected source code.
It never reads repository .env files, run outputs, tests, caches, or vendored
segmentation code.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
from typing import Iterable
from xml.etree import ElementTree
from zipfile import ZipFile


WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
SUBMISSION_LABELS = {
    "Manuscript.docx": ("Manuscript", "paper"),
    "Supplementary.docx": ("Supplementary", "supplement"),
    "Prompt.pdf": ("Prompt design", "supplement"),
    "Algorithm.pdf": ("Algorithm", "supplement"),
    "Table_s1.pdf": ("Supplementary Table S1", "supplement"),
    "Table_s2.pdf": ("Supplementary Table S2", "supplement"),
}
EXCLUDED_CODE_PARTS = {
    ".git",
    ".env",
    "__pycache__",
    "tests",
    "test",
    "demo",
    "results",
    "segmentation_allen",
    "reviewer_knowledge",
    "vector_db",
}
SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
)
LITERAL_SECRET_ASSIGNMENT = re.compile(
    r"(?im)^(\s*(?:export\s+)?[A-Z][A-Z0-9_]*(?:API_KEY|TOKEN|SECRET)\s*=\s*)"
    r"([\"']?)([^\"']+?)\2\s*$"
)


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r", "\n")
    lines = []
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if line:
            lines.append(line)
    return "\n".join(lines)


def _redact_secrets(text: str) -> str:
    def replace_assignment(match: re.Match[str]) -> str:
        prefix, quote = match.group(1), match.group(2)
        return f"{prefix}{quote}[REDACTED]{quote}"

    cleaned = LITERAL_SECRET_ASSIGNMENT.sub(replace_assignment, text)
    for pattern in SECRET_PATTERNS:
        cleaned = pattern.sub("[REDACTED]", cleaned)
    return cleaned


def extract_docx_bytes(data: bytes) -> str:
    """Extract visible paragraphs and table-cell paragraphs from DOCX bytes."""

    with ZipFile(BytesIO(data)) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    paragraphs: list[str] = []
    for paragraph in root.iter(f"{WORD_NS}p"):
        value = "".join(
            node.text or ""
            for node in paragraph.iter(f"{WORD_NS}t")
        ).strip()
        if value:
            paragraphs.append(value)
    return _clean_text("\n".join(paragraphs))


def extract_pdf_bytes(data: bytes) -> str:
    """Extract PDF text with PyMuPDF or the system ``pdftotext`` utility."""

    try:
        import fitz  # type: ignore
    except ImportError:
        executable = shutil.which("pdftotext")
        if not executable:
            return ""
        try:
            completed = subprocess.run(
                [executable, "-", "-"],
                input=data,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        if completed.returncode != 0:
            return ""
        output = completed.stdout
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return _clean_text(str(output))
    document = fitz.open(stream=data, filetype="pdf")
    try:
        pages = [page.get_text("text") for page in document]
    finally:
        document.close()
    return _clean_text("\n".join(pages))


def chunk_text(text: str, *, chunk_chars: int = 3600, overlap_chars: int = 240) -> list[str]:
    """Split text deterministically while keeping paragraph boundaries when possible."""

    value = _clean_text(text)
    if not value:
        return []
    paragraphs = value.split("\n")
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n{paragraph}"
        if len(candidate) <= chunk_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
            prefix = current[-overlap_chars:] if overlap_chars else ""
            current = f"{prefix}\n{paragraph}".strip()
        else:
            current = paragraph
        while len(current) > chunk_chars:
            chunks.append(current[:chunk_chars])
            start = max(0, chunk_chars - overlap_chars)
            current = current[start:]
    if current:
        chunks.append(current)
    return chunks


def _chunk_records(
    *,
    source: str,
    title: str,
    kind: str,
    text: str,
    chunk_chars: int,
) -> list[dict[str, str]]:
    parts = chunk_text(text, chunk_chars=chunk_chars)
    records = []
    for index, part in enumerate(parts, start=1):
        part_title = title if len(parts) == 1 else f"{title} · part {index}"
        records.append(
            {
                "source": source,
                "title": part_title,
                "kind": kind,
                "text": _redact_secrets(part),
            }
        )
    return records


def collect_submission_chunks(
    submission_zip: str | Path,
    *,
    chunk_chars: int = 3600,
) -> list[dict[str, str]]:
    """Extract paper/supplement text from the supplied submission archive."""

    records: list[dict[str, str]] = []
    with ZipFile(Path(submission_zip)) as archive:
        for name in sorted(archive.namelist()):
            path = PurePosixPath(name)
            filename = path.name
            if not filename or "__MACOSX" in path.parts or filename.startswith("._"):
                continue
            if filename in SUBMISSION_LABELS:
                source, kind = SUBMISSION_LABELS[filename]
                data = archive.read(name)
                text = extract_docx_bytes(data) if filename.endswith(".docx") else extract_pdf_bytes(data)
                records.extend(
                    _chunk_records(
                        source=source,
                        title=filename,
                        kind=kind,
                        text=text,
                        chunk_chars=chunk_chars,
                    )
                )
                continue
            if filename.lower().endswith(".pdf") and (
                filename.startswith("Figure_")
                or filename.startswith("Supplementary_Fig_")
            ):
                text = extract_pdf_bytes(archive.read(name))
                records.extend(
                    _chunk_records(
                        source=f"Figure asset: {filename}",
                        title=filename,
                        kind="figure",
                        text=text,
                        chunk_chars=chunk_chars,
                    )
                )
    return records


def _code_file_allowed(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    lowered = {part.lower() for part in relative.parts}
    if lowered & EXCLUDED_CODE_PARTS:
        return False
    if path.name.startswith(".") or path.name.endswith("_generated.py"):
        return False
    return path.suffix == ".py"


def collect_code_chunks(
    code_root: str | Path,
    *,
    chunk_chars: int = 3600,
) -> list[dict[str, str]]:
    """Snapshot relevant first-party Python code without tests, secrets, or outputs."""

    root = Path(code_root).resolve()
    records: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.py")):
        if not _code_file_allowed(path, root):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        relative = path.relative_to(root).as_posix()
        records.extend(
            _chunk_records(
                source=f"Code: {relative}",
                title=relative,
                kind="code",
                text=text,
                chunk_chars=chunk_chars,
            )
        )
    return records


def write_bundle(
    output: str | Path,
    chunks: Iterable[dict[str, str]],
) -> Path:
    path = Path(output)
    records = list(chunks)
    payload = {
        "version": 1,
        "description": "Evidence bundle for Ask MorphAgent reviewer questions.",
        "chunks": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-zip", required=True, type=Path)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk-chars", type=int, default=3600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    chunks = collect_submission_chunks(args.submission_zip, chunk_chars=args.chunk_chars)
    chunks.extend(collect_code_chunks(args.code_root, chunk_chars=args.chunk_chars))
    output = write_bundle(args.output, chunks)
    print(f"Wrote {len(chunks)} reviewer knowledge chunks to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
