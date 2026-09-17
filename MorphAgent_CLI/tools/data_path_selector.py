"""Data path selection tool - intelligently selects the appropriate data source"""
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal
from langchain_core.tools import tool
from config import settings, make_chat_llm


class DataPathSelector:
    """Data path selector - intelligently selects a data source based on feature requirements and the dataset description"""
    
    def __init__(self, verbose: bool = False):
        """Initialize the data path selector
        
        Args:
            verbose: whether to print detailed information (default False; recommended off for batch processing)
        """
        self.llm = make_chat_llm(
            temperature=0,
            max_tokens=200,  # Only need to return path information
        )
        self.verbose = verbose
        self._first_call_info = {}  # Stores information from the first call, used for the summary output
    
    def select_data_paths(
        self,
        sample_dir: Path,
        feature: Dict[str, Any],
        dataset_description: Optional[str] = None,
        method: Literal["code", "vlm"] = "code"
    ) -> List[str] | Dict[str, List[str]]:
        """Intelligently select the appropriate data path based on feature requirements and the dataset description
        
        Data organization:
        - Primary files (raw data): files directly under sample_dir
        - Secondary files (derived data): files in subdirectories under sample_dir
        - Segmentation files: all image files under sample_dir/segmentation/
        
        Selection rules:
        - Code features (code): returns a dict containing 'image_paths' (primary files) and 'segmentation_paths' (segmentation files)
        - VLM features (vlm): prefer secondary files, falling back to primary files if none exist (returns a list of paths)
        
        Args:
            sample_dir: sample directory path
            feature: feature definition
            dataset_description: dataset description (includes an explanation of the data organization)
            method: method type ("code" or "vlm")
            
        Returns:
            For the code method: a dict containing 'image_paths' and 'segmentation_paths'
            For the vlm method: a list of the selected image file paths
        """
        # Scan the available data sources
        available_sources = self._scan_data_sources(sample_dir)
        
        if not available_sources:
            if self.verbose:
                print(f"  [Data Path Selector] ⚠️  No data source found")
            if method == "code":
                return {"image_paths": [], "segmentation_paths": []}
            return []
        
        # Select based on the method type
        if method == "code":
            return self._select_for_code(available_sources, sample_dir, dataset_description)
        else:  # vlm
            return self._select_for_vlm(available_sources, sample_dir, dataset_description)
    
    def _scan_data_sources(self, sample_dir: Path) -> Dict[str, Any]:
        """Scan the sample directory to find all available data sources
        
        Returns:
            A dict containing information about primary and secondary files
        """
        from config import settings
        image_extensions = settings.image_extensions
        
        sources = {
            "primary_files": [],  # Primary files (raw data)
            "secondary_dirs": {}  # Secondary directories (derived data)
        }
        
        # Scan primary files
        for item in sample_dir.iterdir():
            if item.is_file() and item.suffix.lower() in image_extensions:
                sources["primary_files"].append({
                    "path": str(item),
                    "name": item.name,
                    "type": "primary_file"
                })
        
        # Scan secondary directories
        for item in sample_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check the image files in the directory
                # For the VLM, prefer PNG/JPG files
                dir_files = []
                for p in item.iterdir():
                    if p.is_file():
                        # Prefer PNG/JPG, but also include other image formats
                        if p.suffix.lower() in image_extensions:
                            dir_files.append(str(p))
                
                if dir_files:
                    # Sort the files: PNG/JPG first
                    def sort_key(path_str):
                        path = Path(path_str)
                        if path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                            return (0, path_str)  # PNG/JPG first
                        else:
                            return (1, path_str)  # Other formats next
                    
                    dir_files_sorted = sorted(dir_files, key=sort_key)
                    sources["secondary_dirs"][item.name] = {
                        "paths": dir_files_sorted,
                        "count": len(dir_files),
                        "type": "secondary_dir"
                    }
        
        return sources
    
    def _select_for_code(
        self,
        available_sources: Dict[str, Any],
        sample_dir: Path,
        dataset_description: Optional[str]
    ) -> Dict[str, List[str]]:
        """Select data paths for code features (images + all segmentation files)
        
        Args:
            available_sources: available data sources
            sample_dir: sample directory
            dataset_description: dataset description
            
        Returns:
            A dict containing 'image_paths' and 'segmentation_paths'
        """
        primary_files = available_sources.get("primary_files", [])
        
        # Select the image file (primary file)
        image_paths = []
        if primary_files:
            # If there are multiple primary files, use the LLM to select the most appropriate one
            if len(primary_files) > 1 and dataset_description:
                selected = self._llm_select_file(primary_files, dataset_description, "code")
                if selected:
                    image_paths = [selected["path"]]
                else:
                    # LLM selection failed, use the first one
                    image_paths = [primary_files[0]["path"]]
            else:
                # Only one file or no description; use it directly
                image_paths = [f["path"] for f in primary_files]
        
        # Use the same scanning logic as main/segmentation (single source of truth)
        from tools.segmentation import list_segmentation_files
        seg_files = list_segmentation_files(sample_dir)
        segmentation_paths = [str(path) for _, path in seg_files]
        
        # Only print detailed information in verbose mode or on the first call
        if self.verbose:
            print(f"  [Data Path Selector] Code feature selection:")
            print(f"    Image files: {len(image_paths)}")
            if image_paths:
                print(f"      Files: {', '.join([Path(p).name for p in image_paths[:3]])}{'...' if len(image_paths) > 3 else ''}")
            print(f"    Segmentation files: {len(segmentation_paths)}")
            if segmentation_paths:
                print(f"      Files: {', '.join([Path(p).name for p in segmentation_paths[:3]])}{'...' if len(segmentation_paths) > 3 else ''}")
        else:
            # Record the information from the first call (for the later summary output)
            key = f"code_{sample_dir.name}"
            if key not in self._first_call_info:
                self._first_call_info[key] = {
                    "method": "code",
                    "image_count": len(image_paths),
                    "segmentation_count": len(segmentation_paths),
                    "image_files": [Path(p).name for p in image_paths[:3]],
                    "segmentation_files": [Path(p).name for p in segmentation_paths[:3]]
                }
        
        return {
            "image_paths": sorted(image_paths),
            "segmentation_paths": segmentation_paths
        }
    
    def _select_for_vlm(
        self,
        available_sources: Dict[str, Any],
        sample_dir: Path,
        dataset_description: Optional[str]
    ) -> List[str]:
        """Select data paths for VLM features.
        
        General priority strategy:
        1. Prefer non-segmentation secondary directories (e.g. slices and other derived image directories);
        2. If there is no suitable secondary directory, use the primary raw image files (e.g. image.tif);
        3. As a last resort, consider directories that are clearly segmentation/masks (e.g. segmentation).
        
        Args:
            available_sources: available data sources
            sample_dir: sample directory
            dataset_description: dataset description
            
        Returns:
            A list of image file paths
        """
        secondary_dirs = available_sources.get("secondary_dirs", {})
        primary_files = available_sources.get("primary_files", [])
        
        # Priority 1: non-segmentation secondary directories (e.g. slices)
        if secondary_dirs:
            # Filter out directories that are clearly segmentation/masks (e.g. segmentation, mask)
            def _is_segmentation_dir(name: str) -> bool:
                lower = name.lower()
                return any(key in lower for key in ["segmentation", "segment", "mask"])
            
            non_seg_dirs = {
                name: info
                for name, info in secondary_dirs.items()
                if not _is_segmentation_dir(name)
            }
            
            if non_seg_dirs:
                # Among the non-segmentation directories, sort by file count descending and pick the first
                sorted_dirs = sorted(
                    non_seg_dirs.items(),
                    key=lambda x: x[1]["count"],
                    reverse=True
                )
                dir_name, dir_info = sorted_dirs[0]
                paths = dir_info["paths"]
                if self.verbose:
                    print(f"  [Data Path Selector] VLM feature selected the {dir_name} directory (secondary files, non-segmentation): {len(paths)} files")
                else:
                    key = f"vlm_{sample_dir.name}"
                    if key not in self._first_call_info:
                        self._first_call_info[key] = {
                            "method": "vlm",
                            "dir_name": dir_name,
                            "file_count": len(paths),
                            "non_segmentation": True,
                        }
                return paths
        
        # Priority 2: primary raw image files (e.g. image.tif)
        if primary_files:
            paths = [f["path"] for f in primary_files]
            if self.verbose:
                print(f"  [Data Path Selector] VLM feature using primary files (raw data): {len(paths)} files")
            else:
                key = f"vlm_{sample_dir.name}"
                if key not in self._first_call_info:
                    self._first_call_info[key] = {
                        "method": "vlm",
                        "file_count": len(paths),
                        "primary": True,
                    }
            return sorted(paths)
        
        # Priority 3: as a fallback, use any secondary directory (including segmentation directories)
        if secondary_dirs:
            # Sort by file count and choose the directory with the most files
            sorted_dirs = sorted(
                secondary_dirs.items(),
                key=lambda x: x[1]["count"],
                reverse=True
            )
            dir_name, dir_info = sorted_dirs[0]
            paths = dir_info["paths"]
            if self.verbose:
                print(f"  [Data Path Selector] VLM feature fallback selected the {dir_name} directory (secondary files, possibly segmentation): {len(paths)} files")
            else:
                key = f"vlm_{sample_dir.name}"
                if key not in self._first_call_info:
                    self._first_call_info[key] = {
                        "method": "vlm",
                        "dir_name": dir_name,
                        "file_count": len(paths),
                        "fallback": True,
                    }
            return paths
        
        if self.verbose:
            print(f"  [Data Path Selector] ⚠️  VLM feature: no data source found")
        return []
    
    def _llm_select_file(
        self,
        files: List[Dict[str, Any]],
        dataset_description: str,
        method: str
    ) -> Optional[Dict[str, Any]]:
        """Use the LLM to select the most appropriate file from multiple files
        
        Args:
            files: list of files
            dataset_description: dataset description
            method: method type ("code" or "vlm")
            
        Returns:
            The selected file dict, or None if it fails
        """
        file_list = "\n".join([f"- {f['name']}" for f in files])
        
        system_prompt = f"""You are a data science expert skilled at selecting the appropriate data file based on feature requirements.

Data organization:
- Primary files (raw data): files directly under the sample directory, such as zstack.tif, MIP.tif
- Secondary files (derived data): files in subdirectories under the sample directory, such as slices/*.png

Selection rules:
- Code features (code): use only primary files (raw data), preferring files that contain the complete data (such as zstack.tif)
- VLM features (vlm): prefer secondary files (such as the slices directory), falling back to primary files if none exist

Based on the dataset description and the file list, select the most appropriate file. Return only the file name, nothing else."""
        
        user_prompt = f"""Dataset description:
{dataset_description[:500]}

Available files:
{file_list}

Method type: {method}

Please select the most appropriate file (return only the file name):"""
        
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            
            selected_name = response.content.strip()
            
            # Find the matching file
            for f in files:
                if selected_name.lower() in f["name"].lower() or f["name"].lower() in selected_name.lower():
                    return f
            
            return None
        except Exception as e:
            if self.verbose:
                print(f"  [Data Path Selector] ⚠️  LLM selection failed: {e}")
            return None


# Global instance
_global_selector = None


def get_data_path_selector(verbose: bool = False) -> DataPathSelector:
    """Get the global data path selector instance (singleton)
    
    Args:
        verbose: whether to print detailed information (default False; recommended off for batch processing)
    """
    global _global_selector
    if _global_selector is None:
        _global_selector = DataPathSelector(verbose=verbose)
    else:
        # Update the verbose setting
        _global_selector.verbose = verbose
    return _global_selector


@tool
def select_data_paths_tool(
    sample_dir: str,
    feature_name: str,
    feature_method: str,
    dataset_description: str = ""
) -> List[str]:
    """Data path selection tool (LangChain Tool)
    
    Intelligently selects the appropriate data path based on feature requirements and method type.
    
    Args:
        sample_dir: sample directory path
        feature_name: feature name
        feature_method: feature method ("code" or "vlm")
        dataset_description: dataset description (optional)
        
    Returns:
        A list of the selected image file paths
    """
    selector = get_data_path_selector()
    sample_path = Path(sample_dir)
    feature = {"name": feature_name, "method": feature_method}
    
    return selector.select_data_paths(
        sample_path,
        feature,
        dataset_description,
        method=feature_method
    )
