"""Tests for the deterministic reviewer-knowledge build tool."""

from __future__ import annotations

import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile


SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "build_reviewer_knowledge.py"


def load_build_module():
    spec = importlib.util.spec_from_file_location("build_reviewer_knowledge", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load build_reviewer_knowledge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReviewerKnowledgeBuildTests(unittest.TestCase):
    def test_secret_redaction_preserves_code_that_reads_environment_variables(self) -> None:
        module = load_build_module()
        source = (
            'DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "")\n'
            'LLM_API_KEY="real-secret-value"\n'
        )

        redacted = module._redact_secrets(source)

        self.assertIn('DEFAULT_LLM_API_KEY = os.getenv("LLM_API_KEY", "")', redacted)
        self.assertIn('LLM_API_KEY="[REDACTED]"', redacted)
        self.assertNotIn("real-secret-value", redacted)

    def test_docx_xml_extraction_keeps_paragraphs_and_tables(self) -> None:
        module = load_build_module()
        document_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
          <w:body>
            <w:p><w:r><w:t>MorphAgent manuscript paragraph.</w:t></w:r></w:p>
            <w:tbl><w:tr><w:tc><w:p><w:r><w:t>Metric</w:t></w:r></w:p></w:tc>
            <w:tc><w:p><w:r><w:t>Value</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
          </w:body>
        </w:document>"""
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr("word/document.xml", document_xml)

        text = module.extract_docx_bytes(buffer.getvalue())

        self.assertIn("MorphAgent manuscript paragraph.", text)
        self.assertIn("Metric", text)
        self.assertIn("Value", text)

    def test_pdf_extraction_falls_back_to_pdftotext_without_pymupdf(self) -> None:
        module = load_build_module()
        completed = SimpleNamespace(returncode=0, stdout="Algorithm evidence\n", stderr="")

        with (
            patch.dict(sys.modules, {"fitz": None}),
            patch("shutil.which", return_value="/opt/homebrew/bin/pdftotext"),
            patch("subprocess.run", return_value=completed) as run,
        ):
            text = module.extract_pdf_bytes(b"%PDF test")

        self.assertEqual(text, "Algorithm evidence")
        run.assert_called_once()

    def test_code_collection_excludes_tests_secrets_and_generated_files(self) -> None:
        module = load_build_module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "tools").mkdir()
            (root / "tests").mkdir()
            (root / "tools" / "feature.py").write_text(
                "def quantify(image):\n    return image.mean()\n",
                encoding="utf-8",
            )
            (root / "tests" / "test_feature.py").write_text("SECRET_TEST = True\n", encoding="utf-8")
            (root / ".env").write_text("LLM_API_KEY=never_bundle_this\n", encoding="utf-8")

            chunks = module.collect_code_chunks(root, chunk_chars=2000)

        joined = "\n".join(chunk["text"] for chunk in chunks)
        self.assertIn("def quantify", joined)
        self.assertNotIn("SECRET_TEST", joined)
        self.assertNotIn("never_bundle_this", joined)
        self.assertTrue(all(chunk["source"].startswith("Code: ") for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
