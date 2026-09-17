"""Tests for lightweight PDF extraction (default RAG path)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from knowledge.pdf_text import extract_text_from_pdf, extract_text_from_pdf_lite


class LitePdfExtractionTests(unittest.TestCase):
    def test_missing_file_returns_error(self) -> None:
        text = extract_text_from_pdf_lite(Path("/tmp/morphagent_missing_pdf_xyz.pdf"))
        self.assertTrue(text.startswith("[ERROR]"))

    def test_default_backend_is_lite(self) -> None:
        with patch("knowledge.pdf_text.extract_text_from_pdf_lite", return_value="hello") as lite:
            with patch.dict("os.environ", {"RAG_PDF_BACKEND": "lite"}, clear=False):
                out = extract_text_from_pdf(Path("dummy.pdf"))
            self.assertEqual(out, "hello")
            lite.assert_called_once()

    def test_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            try:
                import fitz
            except ImportError:
                self.skipTest("pymupdf not installed")
            path = Path(raw) / "t.pdf"
            doc = fitz.open()
            page = doc.new_page()
            # fontsize=1 packs enough glyphs for max_chars truncation.
            page.insert_textbox(page.rect, "alpha " * 20000, fontsize=1)
            doc.save(path)
            doc.close()
            text = extract_text_from_pdf_lite(path, max_chars=200)
            self.assertIn("truncated", text)
            self.assertLessEqual(len(text), 400)


if __name__ == "__main__":
    unittest.main()
