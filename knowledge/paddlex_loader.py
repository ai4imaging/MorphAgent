"""PaddleX PDF Loader - deep document parsing using PaddleX"""
from typing import Iterator, List, Optional
from pathlib import Path
import os

# Try to import paddlex, handling the case where it is not installed
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
    BaseLoader = object
    Document = None


class PaddleXPDFLoader(BaseLoader):
    """
    A Loader for deep document parsing using PaddleX (PP-Structure).

    Suitable for scientific papers; can distinguish Header, Footer, Table, Text, and Image.
    Particularly good at handling multi-column layouts and complex tables.
    """
    
    def __init__(
        self, 
        file_path: str, 
        device: str = "gpu:0", 
        use_layout: bool = True
    ):
        """
        Args:
            file_path: Path to the PDF file
            device: 'cpu', 'gpu:0', 'gpu:1', etc.
            use_layout: Whether to enable layout analysis (recommended True; slower but best for RAG)
        """
        if not PADDLEX_INSTALLED:
            raise ImportError(
                "PaddleX is not installed. Please install it with: "
                "pip install paddlex"
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
        
        # Initialize the PaddleX Pipeline
        # "layout_parsing" is the pipeline specifically designed for complex document parsing
        # Note: model weights are downloaded automatically on the first run
        print(f"  [PaddleX] Initializing the layout_parsing pipeline (device={device})...")
        try:
            self.pipeline = create_pipeline(pipeline="layout_parsing", device=device)
            print(f"  [PaddleX] Pipeline initialized successfully")
        except Exception as e:
            print(f"  [PaddleX] Initialization failed: {e}")
            raise
    
    def load(self) -> List[Document]:
        """Load all documents"""
        return list(self.lazy_load())
    
    def lazy_load(self) -> Iterator[Document]:
        """
        Lazy loading, returning Document objects page by page
        """
        print(f"  [PaddleX] Starting to parse PDF: {self.file_path.name}")
        
        try:
            # PaddleX's predict supports a list or a single file
            output = self.pipeline.predict(str(self.file_path))
            
            page_idx = 0
            for result in output:
                # result corresponds to one page of the PDF or one file
                # The PaddleX output structure contains layout recognition results
                
                page_content = ""
                page_meta = {
                    "source": str(self.file_path),
                    "page_idx": page_idx,
                    "file_name": self.file_path.name
                }
                
                # Process the PaddleX output
                # The PaddleX output format may vary by version and needs flexible handling
                
                # Method 1: Try to get structured output in JSON format
                result_data = None
                if hasattr(result, 'json'):
                    result_data = result.json
                elif hasattr(result, 'to_dict'):
                    result_data = result.to_dict()
                elif isinstance(result, dict):
                    result_data = result
                
                if result_data:
                    # Extract regions (layout analysis results)
                    regions = []
                    if isinstance(result_data, dict):
                        # Possible key names: res, regions, layout, result, etc.
                        for key in ['res', 'regions', 'layout', 'result', 'data']:
                            if key in result_data:
                                regions = result_data[key]
                                if isinstance(regions, dict):
                                    # If regions is a dict, try to extract a list
                                    for sub_key in ['regions', 'items', 'blocks']:
                                        if sub_key in regions:
                                            regions = regions[sub_key]
                                            break
                                break
                    
                    # If regions is a list, process each region
                    if isinstance(regions, list):
                        # Sort by coordinates from top to bottom (if bbox information is available)
                        # Simplified here; process in order of appearance
                        for region in regions:
                            if isinstance(region, dict):
                                category = region.get("type", region.get("category", "")).lower()
                                text = region.get("text", region.get("content", ""))
                                
                                # Strategy: skip headers and footers, keep the rest
                                if category in ["header", "footer"]:
                                    continue
                                
                                # For tables, add a marker
                                if category == "table":
                                    text = f"\n[TABLE CONTENT]\n{text}\n[/TABLE CONTENT]\n"
                                
                                # For titles, add a marker
                                if category == "title":
                                    text = f"\n[TITLE]\n{text}\n[/TITLE]\n"
                                
                                # For figures, keep the caption
                                if category in ["figure", "image"]:
                                    text = f"\n[FIGURE]\n{text}\n[/FIGURE]\n"
                                
                                if text and text.strip():
                                    page_content += text.strip() + "\n"
                
                # Method 2: If result has a direct text attribute
                if not page_content.strip():
                    if hasattr(result, 'text'):
                        page_content = result.text
                    elif hasattr(result, 'content'):
                        page_content = result.content
                    elif isinstance(result, str):
                        page_content = result
                    elif hasattr(result, '__str__'):
                        # Try to convert to a string
                        page_content = str(result)
                
                if page_content.strip():
                    yield Document(
                        page_content=page_content.strip(),
                        metadata=page_meta
                    )
                    page_idx += 1
            
            print(f"  [PaddleX] Parsing complete, {page_idx} pages total")
            
        except Exception as e:
            print(f"  [PaddleX] Parsing error: {e}")
            import traceback
            traceback.print_exc()
            # Return an error document
            yield Document(
                page_content=f"[ERROR] Failed to parse PDF: {e}",
                metadata={"source": str(self.file_path), "error": True}
            )
