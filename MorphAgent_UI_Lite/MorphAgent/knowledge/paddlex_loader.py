"""PaddleX PDF Loader — layout-aware document parsing for scientific papers.

Compatible with PaddleX 3.x (``layout_parsing`` pipeline). Each page is returned
as a LangChain ``Document`` whose ``page_content`` is the clean text reconstructed
from ``parsing_res_list`` (``block_label`` / ``block_content``). Headers and
footers are dropped; tables / titles / figures keep a light marker so the
downstream LLM summary can still see them.
"""
from __future__ import annotations

from typing import Iterator, List, Optional
from pathlib import Path

try:
    from paddlex import create_pipeline
    PADDLEX_INSTALLED = True
except ImportError:
    PADDLEX_INSTALLED = False
    create_pipeline = None

try:
    from langchain_core.document_loaders import BaseLoader
    from langchain_core.documents import Document
    LANGCHAIN_INSTALLED = True
except ImportError:
    LANGCHAIN_INSTALLED = False
    BaseLoader = object  # type: ignore
    Document = None  # type: ignore


_SKIP_LABELS = {"header", "footer", "page_number", "number", "footnote"}


def _result_to_dict(result) -> Optional[dict]:
    """Normalize a PaddleX page result into a plain dict (``res`` payload)."""
    data = None
    if hasattr(result, "json"):
        data = result.json
    elif hasattr(result, "to_dict"):
        data = result.to_dict()
    elif isinstance(result, dict):
        data = result
    if not isinstance(data, dict):
        return None
    # PaddleX 3.x wraps everything under ``res``; older versions put fields at top level.
    return data.get("res", data)


def _blocks_to_text(blocks: List[dict]) -> str:
    """Rebuild page text from a ``parsing_res_list`` of layout blocks."""
    parts: List[str] = []
    for region in blocks:
        if not isinstance(region, dict):
            continue
        label = str(region.get("block_label") or region.get("label") or region.get("type") or "").lower()
        text = region.get("block_content") or region.get("content") or region.get("text") or ""
        if not text or not str(text).strip():
            continue
        text = str(text).strip()
        if label in _SKIP_LABELS:
            continue
        if label in {"table", "table_title"}:
            parts.append(f"\n[TABLE]\n{text}\n[/TABLE]\n")
        elif label in {"title", "paragraph_title", "doc_title"}:
            parts.append(f"\n[TITLE]\n{text}\n[/TITLE]\n")
        elif label in {"figure", "image", "figure_title", "chart"}:
            parts.append(f"\n[FIGURE]\n{text}\n[/FIGURE]\n")
        else:
            parts.append(text)
    return "\n".join(parts).strip()


def _ocr_fallback_text(res: dict) -> str:
    """Fallback: join OCR recognition texts when layout blocks are empty."""
    ocr = res.get("overall_ocr_res") or {}
    texts = ocr.get("rec_texts") or []
    if isinstance(texts, list) and texts:
        return "\n".join(str(t) for t in texts if t).strip()
    return ""


class PaddleXPDFLoader(BaseLoader):
    """Loader for deep document parsing using PaddleX (PP-Structure / layout_parsing).

    Suitable for scientific papers; distinguishes Header, Footer, Table, Text,
    and Image. Particularly good at multi-column layouts and complex tables.
    """

    def __init__(
        self,
        file_path: str,
        device: str = "cpu",
        use_layout: bool = True,
    ):
        """
        Args:
            file_path: Path to the PDF file
            device: 'cpu', 'gpu:0', 'gpu:1', etc.
            use_layout: Kept for API compatibility (always uses layout_parsing)
        """
        if not PADDLEX_INSTALLED:
            raise ImportError(
                "PaddleX is not installed. Please install it with: "
                'pip install "paddlex[ocr]==3.3.10" paddlepaddle==3.0.0'
            )
        if not LANGCHAIN_INSTALLED:
            raise ImportError(
                "LangChain is not installed. Please install it with: "
                "pip install langchain-core"
            )

        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        self.device = device
        self.use_layout = use_layout

        print(f"  [PaddleX] Initializing the layout_parsing pipeline (device={device})...")
        try:
            self.pipeline = create_pipeline(pipeline="layout_parsing", device=device)
            print("  [PaddleX] Pipeline initialized successfully")
        except Exception as e:
            print(f"  [PaddleX] Initialization failed: {e}")
            raise

    def load(self) -> List["Document"]:
        """Load all documents."""
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator["Document"]:
        """Lazy loading, returning Document objects page by page."""
        print(f"  [PaddleX] Starting to parse PDF: {self.file_path.name}")
        try:
            output = self.pipeline.predict(str(self.file_path))
            page_idx = 0
            for result in output:
                res = _result_to_dict(result) or {}
                blocks = res.get("parsing_res_list") or []
                page_content = _blocks_to_text(blocks) if isinstance(blocks, list) else ""
                if not page_content:
                    page_content = _ocr_fallback_text(res)

                if page_content:
                    yield Document(
                        page_content=page_content,
                        metadata={
                            "source": str(self.file_path),
                            "page_idx": page_idx,
                            "file_name": self.file_path.name,
                        },
                    )
                    page_idx += 1

            print(f"  [PaddleX] Parsing complete, {page_idx} pages total")

        except Exception as e:
            print(f"  [PaddleX] Parsing error: {e}")
            import traceback
            traceback.print_exc()
            yield Document(
                page_content=f"[ERROR] Failed to parse PDF: {e}",
                metadata={"source": str(self.file_path), "error": True},
            )
