"""Code executor module - executes code in a sandbox environment and implements ReAct logic"""
import json
import subprocess
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Any, TYPE_CHECKING, Tuple
import sys

from tools.code_gen import CodeGenerator, CodeResult
from tools.code_fix import CodeFixer, CodeFixPlan

if TYPE_CHECKING:
    from state import AgentState
else:
    # Lazy import to avoid circular dependency
    AgentState = Dict[str, Any]


@dataclass
class ExtractionResult:
    """Feature extraction result"""
    values: Dict[str, float]
    errors: Dict[str, str]


def _sandbox_env_name() -> str:
    """Configured conda env used to execute agent-generated feature code."""
    import os
    try:
        from config import settings
        return settings.conda_env
    except Exception:
        return os.getenv("CONDA_ENV", "morphagent_sandbox")


def _find_conda_python(conda_env: str) -> Optional[Path]:
    """Find the Python interpreter path in a conda environment"""
    import os
    
    conda_base = None
    
    # Check the CONDA_BASE_PATH environment variable
    if "CONDA_BASE_PATH" in os.environ:
        conda_base = Path(os.environ["CONDA_BASE_PATH"])
        if not conda_base.exists():
            conda_base = None
    
    # Detect from CONDA_PREFIX
    if not conda_base and "CONDA_PREFIX" in os.environ:
        current_env = os.environ.get("CONDA_PREFIX", "")
        if current_env:
            current_path = Path(current_env)
            if "envs" in current_path.parts:
                idx = current_path.parts.index("envs")
                conda_base = Path(*current_path.parts[:idx])
    
    # Try common conda installation paths (read from config)
    if not conda_base or not conda_base.exists():
        from config import settings
        for base_path in settings.conda_base_paths:
            if base_path.exists():
                conda_base = base_path
                break
    
    if conda_base:
        conda_env_path = conda_base / "envs" / conda_env
        if conda_env_path.exists():
            # Try to find the Python executable (read the version list from config)
            from config import settings
            for python_name in settings.python_versions:
                python_path = conda_env_path / "bin" / python_name
                if python_path.exists() and python_path.is_file():
                    return python_path
    
    # Fallback: try using conda run
    try:
        result = subprocess.run(
            ["conda", "run", "-n", conda_env, "which", "python"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            python_path = Path(result.stdout.strip())
            if python_path.exists():
                return python_path
    except Exception:
        pass
    
    return None


def _create_wrapper_script(extract_py_path: Path, data_root: Path, has_segmentation: bool = False, conda_env: Optional[str] = None) -> str:
    """Create a wrapper script used to execute the extract function in the sandbox
    
    Args:
        extract_py_path: path to the extract.py file
        data_root: dataset root directory
        
    Returns:
        wrapper script content
    """
    extract_py_abs_path = extract_py_path.resolve()
    extract_py_dir = extract_py_abs_path.parent
    
    script_lines = [
        "import sys",
        "import json",
        "import traceback",
        "from pathlib import Path",
        "import numpy as np",
        "import os",
        "",
        "# Note: We only pre-import numpy as a minimal convenience.",
        "# All other packages (scipy, skimage, etc.) should be imported inside the extract() function.",
        "# This ensures the code is self-contained and can auto-install missing packages.",
        "",
        "# Import image loading libraries",
        "try:",
        "    import tifffile",
        "except ImportError:",
        "    tifffile = None",
        "",
        "try:",
        "    from PIL import Image",
        "except ImportError:",
        "    Image = None",
        "",
        "try:",
        "    import imageio",
        "except ImportError:",
        "    imageio = None",
        "",
        "try:",
        "    import cv2",
        "except ImportError:",
        "    cv2 = None",
        "",
        "def json_default(value):",
        "    \"\"\"Convert NumPy values returned by generated features to JSON values.\"\"\"",
        "    if isinstance(value, np.generic):",
        "        return value.item()",
        "    if isinstance(value, np.ndarray):",
        "        return value.tolist()",
        "    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')",
        "",
        "# Note: skimage, scipy, and other packages should be imported inside extract() function",
        "# We don't pre-import them here to ensure code is self-contained"
        "",
        "def load_image(image_path):",
        "    \"\"\"Load image from file, handling different formats (TIFF, PNG, JPG, etc.)\"\"\"",
        "    image_path = Path(image_path)",
        "    if not image_path.exists():",
        "        raise FileNotFoundError(f'Image not found: {{image_path}}')",
        "    ",
        "    file_ext = image_path.suffix.lower()",
        "    ",
        "    # TIFF files: use tifffile (primary method)",
        "    if file_ext in ['.tif', '.tiff']:",
        "        if tifffile is None:",
        "            raise ImportError('tifffile is required for TIFF files')",
        "        try:",
        "            img = tifffile.imread(str(image_path))",
        "            return np.asarray(img)",
        "        except Exception as e:",
        "            # If tifffile fails (e.g., not a real TIFF), try PIL as fallback",
        "            if Image is not None:",
        "                try:",
        "                    img = Image.open(str(image_path))",
        "                    return np.array(img)",
        "                except:",
        "                    pass",
        "            raise ValueError(f'Failed to load TIFF file: {{e}}')",
        "    ",
        "    # PNG, JPG, etc.: use PIL (preferred) or imageio",
        "    if Image is not None:",
        "        try:",
        "            img = Image.open(str(image_path))",
        "            arr = np.array(img)",
        "            # Convert RGBA to RGB if needed",
        "            if len(arr.shape) == 3 and arr.shape[2] == 4:",
        "                arr = arr[:, :, :3]",
        "            return arr",
        "        except Exception as e:",
        "            if imageio is None:",
        "                raise ValueError(f'Failed to load image with PIL: {{e}}')",
        "    ",
        "    if imageio is not None:",
        "        try:",
        "            return imageio.imread(str(image_path))",
        "        except Exception as e:",
        "            raise ValueError(f'Failed to load image with imageio: {{e}}')",
        "    ",
        "    raise ImportError('No image loading library available (PIL, imageio, or tifffile)')",
        "",
        "try:",
        "    # Get image path from command line argument",
        "    if len(sys.argv) < 2:",
        "        raise ValueError('Image path required as command line argument')",
        "    image_path_str = sys.argv[1]",
        "    image_path = Path(image_path_str).resolve()",
        "    if not image_path.exists():",
        "        raise FileNotFoundError(f'Image not found: {{image_path}}')",
        "    ",
        "    arr = load_image(image_path)",
        "",
        "    # Load segmentation as key-value dict (key = filename stem, e.g. mask_cell, mask_nucleus)",
        f"    seg = dict()  # seg['mask_cell'], seg['mask_nucleus'], etc. Keys from filenames.",
        f"    if len(sys.argv) >= 3:",
        f"        for i in range(2, len(sys.argv)):",
        f"            seg_path = Path(sys.argv[i]).resolve()",
        f"            if seg_path.exists():",
        f"                key = seg_path.stem  # e.g. mask_cell from mask_cell.tif",
        f"                seg[key] = load_image(seg_path)",
        f"                print(f'Seg loaded {{key!r}} from: {{seg_path.name}}', file=sys.stderr)",
        f"            else:",
        f"                print(f'Warning: Segmentation file not found: {{seg_path}}', file=sys.stderr)",
        f"    # Auto-detect from sample_dir/segmentation/ if no paths passed",
        f"    if len(seg) == 0:",
        f"        image_dir = image_path.parent",
        f"        seg_dir = image_dir / 'segmentation'",
        f"        if seg_dir.exists():",
        f"            _skip = ('visualization', 'visualisation', 'overlay', 'preview', 'rgb', 'color', 'colour', 'summary')",
        f"            for seg_file in sorted(seg_dir.iterdir()):",
        f"                if not seg_file.is_file():",
        f"                    continue",
        f"                if seg_file.suffix.lower() not in {{'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp', '.gif'}}:",
        f"                    continue",
        f"                stem_l = seg_file.stem.lower()",
        f"                if any(tok in stem_l for tok in _skip):",
        f"                    continue",
        f"                key = seg_file.stem",
        f"                seg[key] = load_image(seg_file)",
        f"                print(f'Auto-loaded seg {{key!r}} from: {{seg_file.name}}', file=sys.stderr)",
        f"    # Backward compat: list in sorted key order for extract(img, *segmentation_masks)",
        f"    segmentation_masks = [seg[k] for k in sorted(seg.keys())] if seg else []",
        "",
        "    # Handle different image formats",
        "    # Note: We preserve the original shape for code features",
        "    # The LLM-generated code should handle the data format based on dataset description",
        "    # Only transpose if it's clearly a z-stack (many slices, not channels)",
        "    if arr.ndim == 3:",
        "        # If first dimension is small (< 20), it's likely channels, keep as is",
        "        # If first dimension is large, it might be z-stack, but let code decide",
        "        # For now, preserve original format to match code expectations",
        "        pass  # Keep original shape - code will handle format based on dataset description",
        "",
        "    prepared = arr",
        "",
        "    # Import extract function",
        f"    extract_py_dir = Path({repr(str(extract_py_dir))})",
        "    if str(extract_py_dir) not in sys.path:",
        "        sys.path.insert(0, str(extract_py_dir))",
        "",
        f"    extract_py_path = Path({repr(str(extract_py_abs_path))})",
        "    if not extract_py_path.exists():",
        "        raise FileNotFoundError(f'extract.py not found at {extract_py_path}')",
        "",
        "    # Read and execute extract.py",
        "    with open(extract_py_path, 'r', encoding='utf-8') as f:",
        "        extract_code = f.read()",
        "",
        "    # Execute extract.py - code should import all required packages inside the function",
        "    # We only provide numpy as a minimal convenience, everything else should be imported in extract()",
        "    exec_globals = {",
        "        'np': np,  # Only numpy is pre-imported as a minimal convenience",
        "    }",
        "",
        "    # Auto-install policy (inlined: sandbox env has no MorphAgent package).",
        "    _CORE_SCIENCE = {",
        "        'numpy', 'scipy', 'pandas', 'scikit-image', 'skimage', 'scikit-learn', 'sklearn',",
        "        'opencv-python', 'opencv-python-headless', 'cv2', 'pillow', 'pil', 'matplotlib',",
        "        'tifffile', 'networkx', 'mahotas', 'h5py', 'statsmodels', 'imageio', 'pywavelets',",
        "        'pyyaml', 'yaml', 'tqdm', 'natsort', 'imagecodecs', 'seaborn',",
        "    }",
        "    def _norm_pkg(name):",
        "        return (name or '').strip().lower().replace('_', '-')",
        "    def _is_core_pkg(name):",
        "        top = (name or '').split('.')[0]",
        "        return _norm_pkg(name) in {_norm_pkg(p) for p in _CORE_SCIENCE} or top.lower() in _CORE_SCIENCE",
        "    def auto_install_package(package_name, conda_env=None):",
        "        \"\"\"Try to install a missing non-core package into the code sandbox.\"\"\"",
        "        import re",
        "        import subprocess",
        "        import sys",
        "        name = (package_name or '').strip()",
        "        if not name:",
        "            return False",
        "        if re.search(r'(==|!=|<=|>=|~=|===|<|>|=[0-9])', name):",
        "            print(f'[Auto-install] Blocked version pin: {name!r}', file=sys.stderr)",
        "            return False",
        "        if _is_core_pkg(name):",
        "            print(",
        "                f'[Auto-install] Blocked core package {name!r} — '",
        "                'use the frozen sandbox stack / current APIs (graycomatrix not greycomatrix)',",
        "                file=sys.stderr,",
        "            )",
        "            return False",
        "        try:",
        "            if conda_env:",
        f"                cmd = ['conda', 'run', '-n', conda_env, 'pip', 'install', name]",
        "            else:",
        "                cmd = [sys.executable, '-m', 'pip', 'install', name]",
        "            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)",
        "            if result.returncode == 0:",
        "                return True",
        "            return False",
        "        except Exception:",
        "            return False",
        "",
        "    # Get conda environment for auto-install (UI code sandbox, not the Qt UI env)",
        f"    conda_env_for_install = {repr(conda_env) if conda_env else repr(_sandbox_env_name())}",
        "    if conda_env_for_install is None:",
        f"        conda_env_for_install = os.environ.get('CONDA_ENV', {repr(_sandbox_env_name())})",
        "",
        "    # Try to execute the code, with automatic package installation on ImportError",
        "    max_import_retries = 2",
        "    for retry in range(max_import_retries):",
        "        try:",
        "            code_obj = compile(extract_code, str(extract_py_path), 'exec')",
        "            exec(code_obj, exec_globals)",
        "            break  # Success, exit retry loop",
        "        except ImportError as e:",
        "            import_error_str = str(e)",
        "            package_name = None",
        "            if \"No module named\" in import_error_str:",
        "                import re",
        "                match = re.search(r\"No module named ['\\\"]([^'\\\"]+)['\\\"]\", import_error_str)",
        "                if match:",
        "                    package_name = match.group(1).split('.')[0]",
        "            elif \"cannot import name\" in import_error_str:",
        "                import re",
        "                match = re.search(r\"from ['\\\"]([^'\\\"]+)['\\\"]\", import_error_str)",
        "                if match:",
        "                    package_name = match.group(1).split('.')[0]",
        "",
        "            if package_name and retry < max_import_retries - 1:",
        "                print(f'[Auto-install] Attempting to install missing package: {package_name}', file=sys.stderr)",
        "                if auto_install_package(package_name, conda_env=conda_env_for_install):",
        "                    print(f'[Auto-install] Successfully installed {package_name}, retrying execution...', file=sys.stderr)",
        "                    continue",
        "                else:",
        "                    print(f'[Auto-install] Refused or failed to install {package_name}', file=sys.stderr)",
        "",
        "            exec_error = (",
        "                f'Import error in extract.py: {e}. '",
        "                'Sandbox policy: do not reinstall core science packages; '",
        "                'use current APIs (e.g. graycomatrix not greycomatrix).'",
        "            )",
        "            exec_error_tb = traceback.format_exc()",
        "            raise ValueError(f'{exec_error}\\nTraceback:\\n{exec_error_tb}')",
        "        except SyntaxError as e:",
        "            exec_error = f'Syntax error in extract.py: {e}'",
        "            exec_error_tb = traceback.format_exc()",
        "            raise ValueError(f'{exec_error}\\nTraceback:\\n{exec_error_tb}')",
        "        except Exception as e:",
        "            exec_error = f'Error executing extract.py: {type(e).__name__}: {e}'",
        "            exec_error_tb = traceback.format_exc()",
        "            raise ValueError(f'{exec_error}\\nTraceback:\\n{exec_error_tb}')",
        "",
        "    # Get extract function (try extract_all first for merged code, then extract)",
        "    extract_func = exec_globals.get('extract_all', None)",
        "    if extract_func is None:",
        "        extract_func = exec_globals.get('extract', None)",
        "    if extract_func is None:",
        "        available_names = [k for k in exec_globals.keys() if not k.startswith('_')]",
        "        error_msg = f'extract or extract_all function not found in extract.py. Available names: {available_names}'",
        "        raise ValueError(error_msg)",
        "",
        "    # Execute extraction",
        "    # Try to call with appropriate arguments based on function signature",
        f"    import inspect",
        f"    sig = inspect.signature(extract_func)",
        f"    param_names = list(sig.parameters.keys())",
        f"    num_params = len(param_names)",
        f"    ",
        f"    # Determine which arguments to pass based on function signature",
        f"    # Preferred: extract(img, seg) with seg = dict of name -> array (keys from filenames)",
        f"    if num_params == 1:",
        f"        raw = extract_func(prepared)",
        f"    elif num_params == 2:",
        f"        second = param_names[1]",
        f"        if second in ('seg', 'segmentation') and isinstance(seg, dict):",
        f"            raw = extract_func(prepared, seg)",
        f"        elif len(segmentation_masks) > 0:",
        f"            raw = extract_func(prepared, segmentation_masks[0])",
        f"        else:",
        f"            raw = extract_func(prepared)",
        f"    else:",
        f"        if param_names[1] in ('seg', 'segmentation') and isinstance(seg, dict):",
        f"            raw = extract_func(prepared, seg)",
        f"        elif 'segmentation_masks' in param_names:",
        f"            raw = extract_func(prepared, segmentation_masks)",
        f"        elif len(param_names) == 2 + len(segmentation_masks):",
        f"            args = [prepared] + segmentation_masks",
        f"            raw = extract_func(*args)",
        f"        else:",
        f"            num_mask_params = num_params - 1",
        f"            mask_args = segmentation_masks[:num_mask_params] if len(segmentation_masks) >= num_mask_params else segmentation_masks",
        f"            while len(mask_args) < num_mask_params:",
        f"                mask_args.append(None)",
        f"            raw = extract_func(prepared, *mask_args)",
        "",
        "    # Handle return value: could be float, dict, or other types",
        "    if isinstance(raw, dict):",
        "        # If function returns a dict (merged features), use it directly",
        "        result = raw",
        "    elif isinstance(raw, (int, float)):",
        "        # If function returns a single value, convert to float",
        "        result = float(raw)",
        "    else:",
        "        # Try to convert to float, or use as-is",
        "        try:",
        "            result = float(raw)",
        "        except (ValueError, TypeError):",
        "            result = raw",
        "",
        "    # Output result as JSON",
        "    print(json.dumps({'success': True, 'value': result}, default=json_default))",
        "    sys.exit(0)",
        "",
        "except Exception as e:",
        "    error_msg = traceback.format_exc()",
        "    print(json.dumps({'success': False, 'error': str(e), 'traceback': error_msg}))",
        "    sys.exit(1)",
    ]
    
    return '\n'.join(script_lines)


class CodeExecutor:
    """Code executor - executes code in a sandbox environment"""
    
    def __init__(self, data_root: Path, conda_env: Optional[str] = None, timeout: Optional[int] = None):
        """Initialize the code executor
        
        Args:
            data_root: dataset root directory
            conda_env: conda environment name (defaults to reading from config)
            timeout: timeout in seconds (defaults to reading from config)
        """
        from config import settings
        self.data_root = data_root
        self.conda_env = conda_env or settings.conda_env
        self.timeout = timeout or settings.code_sandbox_timeout
        self._python_path = None
    
    def _get_python_path(self) -> Path:
        """Get the Python interpreter path"""
        if self._python_path is None:
            if self.conda_env:
                python_path = _find_conda_python(self.conda_env)
                if python_path is None:
                    raise RuntimeError(
                        f"Unable to find Python interpreter in conda environment '{self.conda_env}'. "
                        f"Please ensure the environment exists and is accessible."
                    )
                self._python_path = python_path
            else:
                # Use the current Python
                self._python_path = Path(sys.executable)
        return self._python_path
    
    def execute_single_sample(
        self,
        extract_py_path: Path,
        image_path: Path,
        segmentation_paths: Optional[List[Path]] = None
    ) -> tuple[bool, Optional[float], Optional[str]]:
        """Execute feature extraction on a single sample
        
        Args:
            extract_py_path: path to the extract.py file
            image_path: path to the image file
            
        Returns:
            (success, value, error_message) tuple
        """
        python_path = self._get_python_path()
        
        # Create the wrapper script
        feature_dir = extract_py_path.parent
        wrapper_script_path = feature_dir / "runner.py"
        
        # Check whether the wrapper script needs to be updated
        wrapper_needs_update = False
        if not wrapper_script_path.exists():
            wrapper_needs_update = True
        else:
            # If extract.py was updated, the wrapper needs to be regenerated
            extract_mtime = extract_py_path.stat().st_mtime
            wrapper_mtime = wrapper_script_path.stat().st_mtime
            if extract_mtime > wrapper_mtime:
                wrapper_needs_update = True
        
        # Prepare the list of segmentation paths (converted to strings)
        seg_paths_str = []
        if segmentation_paths:
            for seg_path in segmentation_paths:
                seg_path_obj = Path(seg_path) if not isinstance(seg_path, Path) else seg_path
                if seg_path_obj.exists():
                    seg_paths_str.append(str(seg_path_obj.resolve()))
        
        if wrapper_needs_update:
            # No longer pass image_path, since it is now read from command-line arguments
            wrapper_content = _create_wrapper_script(extract_py_path, self.data_root, len(seg_paths_str) > 0, conda_env=self.conda_env)
            wrapper_script_path.write_text(wrapper_content, encoding='utf-8')
        
        try:
            # Execute the wrapper script, passing the image path and all segmentation paths as command-line arguments
            image_path_abs = image_path.resolve()
            cmd = [str(python_path), str(wrapper_script_path), str(image_path_abs)]
            # Add all segmentation paths
            cmd.extend(seg_paths_str)
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            
            # Parse the output
            if result.returncode == 0:
                try:
                    output = json.loads(result.stdout.strip())
                    if output.get("success"):
                        return True, output.get("value"), None
                    else:
                        error_msg = output.get("error", "Unknown error")
                        return False, None, error_msg
                except json.JSONDecodeError:
                    # If not JSON, try to parse directly as a float
                    try:
                        value = float(result.stdout.strip())
                        return True, value, None
                    except ValueError:
                        return False, None, f"Failed to parse output: {result.stdout}"
            else:
                # Try to parse the error from stderr or stdout
                error_output = result.stderr or result.stdout
                try:
                    error_json = json.loads(error_output.strip())
                    error_msg = error_json.get("error", "Unknown error")
                    traceback_info = error_json.get("traceback", "")
                    if traceback_info:
                        # Include the full traceback information, which is important for debugging
                        full_error = f"{error_msg}\n\nTraceback:\n{traceback_info}"
                    else:
                        full_error = error_msg
                    return False, None, full_error
                except json.JSONDecodeError:
                    # If not in JSON format, try to extract the full error message from stderr and stdout
                    error_msg = error_output if error_output else "Unknown error"
                    # If both stderr and stdout have content, merge them
                    if result.stderr and result.stdout and result.stderr != result.stdout:
                        error_msg = f"{result.stderr}\n{result.stdout}"
                    return False, None, f"Execution failed (return code {result.returncode}): {error_msg}"
        
        except subprocess.TimeoutExpired:
            return False, None, f"Execution timeout (exceeded {self.timeout} seconds)"
        except Exception as exc:
            return False, None, f"Execution exception: {str(exc)}"
    
    def execute_all_samples(
        self,
        extract_py_path: Path,
        sample_ids: List[str],
        find_image_paths_func,
        log_file: Optional[Path] = None,
        segmentation_mask_path: Optional[Path] = None
    ) -> ExtractionResult:
        """Execute feature extraction on all samples
        
        Args:
            extract_py_path: path to the extract.py file
            sample_ids: list of sample IDs
            find_image_paths_func: function to find image paths
            log_file: path to the log file (optional)
            
        Returns:
            ExtractionResult object
        """
        values = {}
        errors = {}
        
        # Open the log file (if provided)
        log_fp = None
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_fp = open(log_file, 'a', encoding='utf-8')
        
        try:
            # Use tqdm to display a progress bar
            from tqdm import tqdm
            pbar = tqdm(sample_ids, desc="  Processing samples", leave=False, ncols=80)
            for sample_id in pbar:
                sample_dir = self.data_root / sample_id
                image_paths = find_image_paths_func(sample_dir, "")
                
                if not image_paths:
                    error_msg = "Image file not found"
                    errors[sample_id] = error_msg
                    if log_fp:
                        log_fp.write(f"{sample_id}: ❌ {error_msg}\n")
                    tqdm.write(f"    [Code] {sample_id}: ❌ {error_msg}")
                    continue
                
                # Get the image path and segmentation paths
                # find_image_paths_func now returns a list of image paths, but we need the full path information
                sample_dir = self.data_root / sample_id
                
                # Call the selector again to get the full path information (including segmentation)
                from tools.data_path_selector import get_data_path_selector
                selector = get_data_path_selector(verbose=False)
                path_result = selector.select_data_paths(
                    sample_dir,
                    {"method": "code"},
                    None,  # dataset_description can be None, since scanning was already done
                    method="code"
                )
                
                if isinstance(path_result, dict):
                    image_paths_full = path_result.get("image_paths", [])
                    segmentation_paths = path_result.get("segmentation_paths", [])
                else:
                    # Backward compatibility with the old format
                    image_paths_full = image_paths if isinstance(path_result, list) else []
                    segmentation_paths = []
                
                if not image_paths_full:
                    error_msg = "Image file not found"
                    errors[sample_id] = error_msg
                    if log_fp:
                        log_fp.write(f"{sample_id}: ❌ {error_msg}\n")
                    tqdm.write(f"    [Code] {sample_id}: ❌ {error_msg}")
                    continue
                
                # Use the first image
                image_path = Path(image_paths_full[0])
                
                # Print debug information (use tqdm.write to avoid interrupting the progress bar)
                tqdm.write(f"    [Code] {sample_id}: Image path = {image_path}")
                if segmentation_paths:
                    tqdm.write(f"    [Code] {sample_id}: Segmentation files = {len(segmentation_paths)} ({', '.join([Path(p).name for p in segmentation_paths[:3]])}{'...' if len(segmentation_paths) > 3 else ''})")
                
                success, value, error_msg = self.execute_single_sample(
                    extract_py_path, image_path, segmentation_paths
                )
                
                if success:
                    values[sample_id] = value
                    if log_fp:
                        log_fp.write(f"{sample_id}: ✅ {value} (image: {image_path})\n")
                    tqdm.write(f"    [Code] {sample_id}: ✅ Result = {value}")
                else:
                    errors[sample_id] = error_msg or "Unknown error"
                    if log_fp:
                        log_fp.write(f"{sample_id}: ❌ {error_msg} (image: {image_path})\n")
                        log_fp.flush()  # Ensure it is written immediately
                    # Show the full error message (not truncated), but limit the display length
                    error_display = error_msg if len(error_msg) <= 500 else error_msg[:500] + "..."
                    tqdm.write(f"    [Code] {sample_id}: ❌ {error_display}")
        finally:
            if log_fp:
                log_fp.close()
        
        return ExtractionResult(values=values, errors=errors)


def run_code_generation_with_react(
    feature: Dict[str, Any],
    state: AgentState,
    sample_ids: List[str],
    data_root: Path,
    find_image_paths_func,
    results_dir: Optional[Path] = None,
    conda_env: Optional[str] = None,
    max_cycles: Optional[int] = None,
    segmentation_mask_path: Optional[Path] = None,
    enable_critic: Optional[bool] = None
) -> tuple[ExtractionResult, CodeResult]:
    """Run code generation and execution, including ReAct logic
    
    Args:
        feature: feature definition
        state: Agent state
        sample_ids: list of sample IDs
        data_root: dataset root directory
        find_image_paths_func: function to find image paths
        results_dir: directory to save results (optional)
        conda_env: conda environment name (defaults to reading from config)
        max_cycles: maximum number of retries (defaults to reading from config)
        
    Returns:
        (ExtractionResult, CodeResult) tuple
    """
    from config import settings
    max_cycles = max_cycles or settings.code_max_retries
    generator = CodeGenerator()
    fixer = CodeFixer()
    executor = CodeExecutor(data_root, conda_env=conda_env)
    
    # Create the feature directory
    feature_name = feature.get("name", "unknown")
    if results_dir:
        feature_dir = results_dir / "features" / feature_name
        feature_dir.mkdir(parents=True, exist_ok=True)
    else:
        feature_dir = Path(f"/tmp/morphagent_features/{feature_name}")
        feature_dir.mkdir(parents=True, exist_ok=True)
    
    extract_py_path = feature_dir / "extract.py"
    guidance_message = ""
    code_result = CodeResult(code=None, prompt="", response="")
    
    print(f"\n[Code Execution] Starting to process feature '{feature_name}'")
    print(f"  Maximum retry count: {max_cycles}")
    
    # Before generating code, collect data statistics (from the first sample)
    data_statistics = {}
    planning_text = None
    
    if sample_ids:
        first_sample_id = sample_ids[0]
        first_sample_dir = data_root / first_sample_id
        
        # Get the full path information (image and segmentation)
        from tools.data_path_selector import get_data_path_selector
        selector = get_data_path_selector(verbose=False)
        path_result = selector.select_data_paths(
            first_sample_dir,
            {"method": "code"},
            state.get("research_summary", ""),
            method="code"
        )
        
        first_image_paths = []
        first_seg_paths = []
        if isinstance(path_result, dict):
            first_image_paths = path_result.get("image_paths", [])
            first_seg_paths = path_result.get("segmentation_paths", [])
        
        # Collect data statistics
        if first_image_paths:
            from tools.data_statistics import collect_data_statistics
            image_path = Path(first_image_paths[0])
            seg_paths = [Path(p) for p in first_seg_paths] if first_seg_paths else []
            
            print(f"  [Data Statistics] Collecting data statistics...")
            data_statistics = collect_data_statistics(
                image_path=image_path,
                segmentation_paths=seg_paths,
                dataset_description=state.get("research_summary", ""),
                sample_dir=first_sample_dir,
            )
            
            # If the state has no mask order information, try to read it from the segmentation summary file
            if not state.get("segmentation_mask_order") and data_root:
                try:
                    # Try to read segmentation_summary.json from the results directory
                    results_dir = data_root.parent / "results" if "results" in str(data_root.parent) else None
                    if results_dir and results_dir.exists():
                        # Find the latest segmentation_summary.json
                        import json
                        import glob
                        summary_files = sorted(glob.glob(str(results_dir / "**/segmentation_summary.json")), reverse=True)
                        if summary_files:
                            with open(summary_files[0], 'r', encoding='utf-8') as f:
                                summary = json.load(f)
                                if summary.get("mask_order_description"):
                                    state["segmentation_mask_order"] = summary["mask_order_description"]
                                    print(f"  ✅ Read mask order information from the segmentation summary")
                except Exception as e:
                    print(f"  ⚠️  Failed to read the segmentation summary: {e}")
            
            # If still not available, use the one generated in data_statistics
            if not state.get("segmentation_mask_order") and data_statistics.get("segmentation_mask_order"):
                state["segmentation_mask_order"] = data_statistics["segmentation_mask_order"]
            
            state["data_statistics"] = data_statistics
            print(f"  ✅ Statistics collection completed")
            
            # Generate the CoT plan
            print(f"\n  [CoT Planning] Generating the code implementation plan...")
            planning_text = generator.generate_planning(
                feature=feature,
                state=state,
                data_statistics=data_statistics
            )
            if planning_text:
                print(f"  ✅ Plan generation completed")
                # Save the plan
                if results_dir:
                    planning_path = feature_dir / "code_planning.txt"
                    with open(planning_path, 'w', encoding='utf-8') as f:
                        f.write(planning_text)
                    print(f"  💾 Plan saved: {planning_path}")
            else:
                print(f"  ⚠️  Plan generation failed; code will be generated directly")
    
    for attempt in range(1, max_cycles + 1):
        print(f"\n  Attempt {attempt}/{max_cycles}")
        
        # Generate code (passing in the CoT plan)
        code_result = generator.generate(
            feature, 
            state, 
            guidance_message=guidance_message,
            planning_text=planning_text if attempt == 1 else None  # Only use the CoT plan on the first attempt
        )
        
        # Save the prompt and response
        if results_dir:
            prompt_path = feature_dir / (
                "code_prompt.json" if attempt == 1 else f"code_prompt_retry_{attempt - 1}.json"
            )
            with open(prompt_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "prompt": code_result.prompt,
                    "response": code_result.response,
                    "planning_response": code_result.planning_response if hasattr(code_result, 'planning_response') else None
                }, f, indent=2, ensure_ascii=False)
        
        if not code_result.code:
            print(f"    ❌ Failed to generate valid code")
            if attempt == max_cycles:
                return ExtractionResult(values={}, errors={sid: "Code generation failed" for sid in sample_ids}), code_result
            continue
        
        # Save code
        extract_py_path.write_text(code_result.code, encoding='utf-8')
        print(f"    ✅ Code saved: {extract_py_path}")
        
        # First test the first sample, and get segmentation file information for the prompt
        segmentation_files_info = ""
        if sample_ids:
            first_sample_id = sample_ids[0]
            first_sample_dir = data_root / first_sample_id
            
            # Get the list of segmentation files
            from tools.data_path_selector import get_data_path_selector
            selector = get_data_path_selector(verbose=False)
            path_result = selector.select_data_paths(
                first_sample_dir,
                {"method": "code"},
                None,
                method="code"
            )
            
            if isinstance(path_result, dict):
                first_seg_paths = path_result.get("segmentation_paths", [])
                if first_seg_paths:
                    segmentation_files_info = "\n**Available Segmentation Files** (in `sample_dir/segmentation/`):\n"
                    for i, seg_path in enumerate(first_seg_paths, 1):
                        seg_file = Path(seg_path)
                        segmentation_files_info += f"  {i}. `{seg_file.name}` (stem: `{seg_file.stem}`)\n"
                    segmentation_files_info += f"\nTotal: {len(first_seg_paths)} segmentation file(s) available.\n"
            
            # Get the full path information for the first sample (including segmentation)
            from tools.data_path_selector import get_data_path_selector
            selector = get_data_path_selector(verbose=False)
            path_result = selector.select_data_paths(
                first_sample_dir,
                {"method": "code"},
                state.get("research_summary", ""),
                method="code"
            )
            
            first_image_paths = []
            first_seg_paths = []
            if isinstance(path_result, dict):
                first_image_paths = path_result.get("image_paths", [])
                first_seg_paths = path_result.get("segmentation_paths", [])
            else:
                # Backward compatibility with the old format
                first_image_paths = find_image_paths_func(first_sample_dir, "")
            
            if first_image_paths:
                first_image_path = Path(first_image_paths[0])
                # Convert segmentation paths to Path objects
                first_seg_paths_objs = [Path(p) for p in first_seg_paths] if first_seg_paths else []
                success, value, error_msg = executor.execute_single_sample(
                    extract_py_path, first_image_path, first_seg_paths_objs
                )
                
                if not success:
                    # Show the full error message for debugging
                    error_display = error_msg if len(error_msg) <= 1000 else error_msg[:1000] + "..."
                    print(f"    ❌ First sample test failed: {error_display}")
                    
                    # Check if it's a timeout error
                    if "timeout" in error_msg.lower() or "timed out" in error_msg:
                        print(f"    ⚠️  Timeout error detected")
                        if attempt == max_cycles:
                            return ExtractionResult(
                                values={},
                                errors={sid: f"Execution timeout: {error_msg}" for sid in sample_ids}
                            ), code_result
                        
                        # Generate fix plan
                        timeout_error = {"timeout": error_msg}
                        fix_plan = fixer.plan(feature, code_result.code, timeout_error)
                        if fix_plan:
                            _handle_code_fix(fix_plan, feature_dir, feature_name, conda_env=conda_env)
                            guidance_message = fix_plan.guidance_message
                        else:
                            guidance_message = "Code execution timed out. Please optimize the algorithm to reduce computation time, avoid nested loops over large arrays, and use vectorized operations."
                        continue
                    
                    # Other errors, generate fix plan
                    if attempt < max_cycles:
                        error_map = {first_sample_id: error_msg}
                        print(f"    🔧 Generating fix plan for error: {error_msg[:300]}...")
                        fix_plan = fixer.plan(feature, code_result.code, error_map)
                        if fix_plan:
                            print(f"    ✅ Fix plan generated")
                            print(f"       Install script: {fix_plan.install_script[:100] if fix_plan.install_script else 'None'}...")
                            print(f"       Guidance: {fix_plan.guidance_message[:200] if fix_plan.guidance_message else 'None'}...")
                            _handle_code_fix(fix_plan, feature_dir, feature_name, conda_env=conda_env)
                            guidance_message = fix_plan.guidance_message
                        else:
                            print(f"    ⚠️  Failed to generate fix plan")
                            guidance_message = f"Previous error: {error_msg[:500]}"
                        continue
                    else:
                        # The last attempt failed
                        return ExtractionResult(
                            values={},
                            errors={sid: error_msg for sid in sample_ids}
                        ), code_result
                else:
                    print(f"    ✅ First sample test succeeded: {value}")
        
        # The first sample passed; execute all samples
        # Create the log file
        log_file = None
        if results_dir:
            log_file = feature_dir / "execution_log.txt"
        
        result = executor.execute_all_samples(
            extract_py_path, 
            sample_ids, 
            find_image_paths_func,
            log_file=log_file,
            segmentation_mask_path=segmentation_mask_path
        )
        
        if not result.errors:
            print(f"    ✅ All samples executed successfully ({len(result.values)}/{len(sample_ids)})")
            if log_file:
                print(f"    Detailed results saved to: {log_file}")
            
            # Check if all values are 0.0 (usually indicates a code problem)
            if result.values:
                all_zero = all(abs(v) < 1e-10 for v in result.values.values())
                if all_zero and attempt < max_cycles:
                    print(f"    ⚠️  Detected all feature values are 0.0, which usually indicates a logic error in the code")
                    print(f"    Triggering ReAct fix mechanism...")
                    
                    # Build an error message explaining that all values are 0.0
                    zero_error = {
                        "all_samples": f"All {len(result.values)} samples returned 0.0. This usually indicates a logic error in the code, such as: incorrect channel selection, wrong array indexing, incorrect data type handling, or a bug in the calculation formula. Please review the code and fix the issue."
                    }
                    
                    fix_plan = fixer.plan(feature, code_result.code, zero_error)
                    if fix_plan:
                        print(f"    ✅ Fix plan generated for all-zero values")
                        print(f"       Guidance: {fix_plan.guidance_message[:200] if fix_plan.guidance_message else 'None'}...")
                        _handle_code_fix(fix_plan, feature_dir, feature_name, conda_env=conda_env)
                        guidance_message = fix_plan.guidance_message
                        print(f"    🔄 Retrying with fix guidance...")
                        continue
                    else:
                        print(f"    ⚠️  Failed to generate fix plan for all-zero values")
                        guidance_message = "All samples returned 0.0. Please check: (1) Are you using the correct channel/index? (2) Is the array indexing correct? (3) Is the calculation formula correct? (4) Are there any data type issues?"
                        continue
            
            return result, code_result
        
        # There are errors, but the first sample passed, might be special cases for some samples
        print(f"    ⚠️  Some samples failed: {len(result.errors)}/{len(sample_ids)}")
        
        # If there are too many errors, try to fix them
        error_rate = len(result.errors) / len(sample_ids)
        from config import settings
        error_threshold = settings.code_error_rate_threshold
        if error_rate > error_threshold and attempt < max_cycles:  # Failure rate exceeds the threshold
            print(f"    🔧 Error rate ({error_rate:.1%}) exceeds threshold ({error_threshold:.1%}), generating fix plan...")
            fix_plan = fixer.plan(feature, code_result.code, result.errors)
            if fix_plan:
                print(f"    ✅ Fix plan generated")
                print(f"       Guidance: {fix_plan.guidance_message[:200] if fix_plan.guidance_message else 'None'}...")
                _handle_code_fix(fix_plan, feature_dir, feature_name, conda_env=conda_env)
                guidance_message = fix_plan.guidance_message
                print(f"    🔄 Retrying with fix guidance...")
                continue
            else:
                print(f"    ⚠️  Failed to generate fix plan")
                # Use partial error information as guidance
                sample_errors = list(result.errors.items())[:3]  # Take the first 3 errors
                error_summary = "\n".join([f"{sid}: {err[:200]}" for sid, err in sample_errors])
                guidance_message = f"Previous errors (sample):\n{error_summary}"
                continue
        
        # Return the partially successful result
        return result, code_result
    
    # All attempts failed
    print(f"    ❌ Reached maximum retry count, giving up on this feature")
    return ExtractionResult(
        values={},
        errors={sid: "Reached maximum retry count, code execution failed" for sid in sample_ids}
    ), code_result


def _handle_code_fix(plan: CodeFixPlan, feature_dir: Path, feature_name: str, conda_env: Optional[str] = None) -> None:
    """Handle the code fix plan
    
    Args:
        plan: fix plan
        feature_dir: feature directory
        feature_name: feature name
        conda_env: conda environment name (used to execute the install script in the correct environment)
    """
    from config import settings
    
    # Save the fix plan
    fix_log = feature_dir / "code_fix.json"
    with open(fix_log, 'w', encoding='utf-8') as f:
        json.dump({
            "prompt": plan.prompt,
            "response": plan.response,
            "install_script": plan.install_script,
            "guidance_message": plan.guidance_message
        }, f, indent=2, ensure_ascii=False)
    
    if plan.guidance_message:
        (feature_dir / "code_fix_guidance.txt").write_text(plan.guidance_message, encoding="utf-8")
        print(f"    📝 Fix guidance: {plan.guidance_message[:200]}")
    
    # Execute the install script
    if plan.install_script:
        from tools.sandbox_install_policy import (
            blocked_install_guidance,
            validate_install_script,
        )

        ok, reason, script = validate_install_script(plan.install_script)
        if not ok:
            print(f"    🚫 Sandbox install blocked: {reason}")
            guidance = blocked_install_guidance(reason)
            existing = (plan.guidance_message or "").strip()
            plan.guidance_message = (existing + "\n\n" + guidance).strip() if existing else guidance
            (feature_dir / "code_fix_guidance.txt").write_text(plan.guidance_message, encoding="utf-8")
            (feature_dir / "code_fix_install_blocked.txt").write_text(
                f"blocked: {reason}\n\noriginal_script:\n{plan.install_script}\n",
                encoding="utf-8",
            )
            script = ""
        else:
            script = script.strip()

        if not script:
            return

        # Get the target conda environment
        target_env = conda_env or settings.conda_env
        
        # If the script does not specify a conda environment and the target environment exists, wrap the script to execute in the correct environment
        if target_env and ("conda run" not in script.lower() and "conda activate" not in script.lower()):
            # Check whether the script contains a pip install or conda install command
            if "pip install" in script or "conda install" in script or "pip3 install" in script:
                # Wrap the script to execute in the specified environment using conda run
                wrapped_script = f"""#!/usr/bin/env bash
set -e
# Execute the install command in conda environment {target_env}
conda run -n {target_env} bash << 'INSTALL_EOF'
{script}
INSTALL_EOF
"""
                script = wrapped_script
                print(f"    🔧 Wrapping install script to execute in conda environment '{target_env}'")
        
        if not script.startswith("#!"):
            script = "#!/usr/bin/env bash\n" + script
        
        script_path = feature_dir / "code_fix_install.sh"
        script_path.write_text(script + "\n", encoding="utf-8")
        script_path.chmod(0o755)
        
        try:
            subprocess.run(["bash", str(script_path)], check=True, timeout=60)
            print(f"    ✅ Fix script executed successfully (in environment '{target_env}')")
        except subprocess.CalledProcessError as exc:
            print(f"    ⚠️  Fix script execution failed: {exc}")
        except subprocess.TimeoutExpired:
            print(f"    ⚠️  Fix script execution timeout")


def evaluate_result_with_critic(
    feature: Dict[str, Any],
    image_path: Path,
    result_value: float,
    code: str,
    results_dir: Optional[Path] = None,
    feature_dir: Optional[Path] = None,
    enable_critic: Optional[bool] = None
) -> Tuple[bool, str]:
    """Use a Critic Agent (VLM) to evaluate the reasonableness of the feature extraction result
    
    Args:
        feature: feature definition
        image_path: path to the test image
        result_value: extracted feature value
        code: extraction code
        results_dir: directory to save results (optional)
        feature_dir: feature directory (optional, used to save the critic evaluation result)
        enable_critic: whether to enable the critic agent (optional; if None, read from settings)
        
    Returns:
        (passed: bool, feedback: str) tuple
        - passed: whether the evaluation passed
        - feedback: evaluation feedback (if not passed, includes improvement suggestions)
    """
    from config import settings
    
    # Check whether the critic agent is enabled
    if enable_critic is None:
        enable_critic = settings.enable_critic_agent
    
    if not enable_critic:
        print(f"      [Critic] Critic agent is disabled; skipping evaluation")
        return True, ""
    
    print(f"      [Critic] Using the VLM to evaluate result reasonableness...")
    
    try:
        import re
        from tools.vlm_client import get_vlm_client
        
        # Get the VLM client
        vlm_client = get_vlm_client()
        
        # Build the critic evaluation prompt
        feature_name = feature.get("name", "unknown")
        feature_description = feature.get("description", "")
        
        # Check whether the result is obviously unreasonable
        is_zero = abs(result_value) < 1e-10
        is_negative = result_value < 0
        is_extreme = abs(result_value) > 1e10  # Extremely large value
        
        # Build the evaluation prompt
        critic_prompt = f"""You are a scientific image analysis expert acting as a critic agent. Your task is to evaluate whether a feature extraction result is reasonable.

**Feature Information:**
- Feature Name: {feature_name}
- Feature Description: {feature_description}

**Extraction Result:**
- Result Value: {result_value}
- Is Zero: {is_zero}
- Is Negative: {is_negative}
- Is Extreme: {is_extreme}

**Extraction Code:**
```python
{code}
```

**Task:**
Please examine the input image and evaluate whether the extraction result is reasonable based on:
1. The feature description and what it should measure
2. The visual content of the image
3. Whether the result value makes sense (e.g., area should be positive, ratio should be reasonable, etc.)

**Output Format:**
Please respond in the following JSON format:
{{
    "passed": true/false,
    "reason": "brief explanation of why it passed or failed",
    "feedback": "if failed, provide specific guidance on what might be wrong and how to fix it"
}}

**Common Issues to Check:**
- Result is 0 when it should be non-zero (e.g., area, intensity, count)
- Result is negative when it should be positive (e.g., area, length, count)
- Result is unreasonably large or small
- Result doesn't match what the feature description suggests
- Code might be using wrong channel, wrong mask, or incorrect calculation

Please analyze the image and provide your evaluation."""
        
        # Prepare the image path (convert to PNG format if the VLM requires it)
        # The VLM client automatically handles image format conversion
        image_paths = [str(image_path.resolve())]

        def _save_critic_result(passed: bool, feedback: str, full_response: str, reason: str = "") -> None:
            if feature_dir:
                critic_result_path = feature_dir / "critic_evaluation.json"
                with open(critic_result_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "result_value": result_value,
                        "passed": passed,
                        "reason": reason,
                        "feedback": feedback,
                        "full_response": full_response
                    }, f, indent=2, ensure_ascii=False)

        def _parse_critic_response(generated_text: str) -> Tuple[bool, str]:
            json_match = re.search(r'\{[^{}]*"passed"[^{}]*\}', generated_text, re.DOTALL)
            if json_match:
                try:
                    critic_result = json.loads(json_match.group(0))
                    passed = critic_result.get("passed", True)
                    reason = critic_result.get("reason", "")
                    feedback = critic_result.get("feedback", reason)
                    _save_critic_result(passed, feedback, generated_text, reason)

                    if passed:
                        print(f"      ✅ Critic evaluation passed: {reason or 'Result is reasonable'}")
                    else:
                        print(f"      ❌ Critic evaluation not passed: {reason or 'Result is unreasonable'}")
                    return passed, feedback
                except json.JSONDecodeError:
                    pass

            if "passed" in generated_text.lower() and "false" in generated_text.lower():
                feedback = generated_text
                _save_critic_result(False, feedback, generated_text, "")
                return False, feedback

            print(f"      ⚠️  Critic response format could not be parsed; passing by default")
            _save_critic_result(True, "", generated_text, "")
            return True, ""
        
        # Call the VLM to perform the evaluation
        try:
            vlm_client._load_model()

            if settings.vlm_api_provider == "online":
                processed_image_paths = vlm_client._preprocess_images(image_paths, None)
                image_paths_abs = [str(Path(p).resolve()) for p in processed_image_paths]
                content = vlm_client._images_to_content(image_paths_abs)
                content.append({"type": "text", "text": critic_prompt})
                generated_text = vlm_client._chat_with_retry(content, None)
                if getattr(vlm_client, "_temp_dirs", None):
                    vlm_client.cleanup_temp_files()
                return _parse_critic_response(generated_text)

            # Local Qwen path: retain the original processor/model call logic
            processed_image_paths = vlm_client._preprocess_images(image_paths, None)
            image_paths_abs = [str(Path(p).resolve()) for p in processed_image_paths]

            messages = [{
                "role": "user",
                "content": [
                    *[{"type": "image", "image": path} for path in image_paths_abs],
                    {"type": "text", "text": critic_prompt},
                ],
            }]

            inputs = vlm_client._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )

            import torch
            device = next(vlm_client._model.parameters()).device
            device_inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            del inputs

            with torch.no_grad():
                generated_ids = vlm_client._model.generate(
                    **device_inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=__import__("config").get_vlm_temperature()
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(device_inputs["input_ids"], generated_ids)
            ]

            generated_text = vlm_client._processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            return _parse_critic_response(generated_text)
                
        except Exception as e:
            print(f"      ⚠️  Critic evaluation error (VLM request failed); skipping — NOT counted as a pass: {e}")
            import traceback
            traceback.print_exc()
            # Skip critic on infrastructure / request failures so we do not
            # falsely print "Critic evaluation passed", and do not burn
            # regenerate cycles on a broken VLM call.
            return True, f"[CRITIC_SKIPPED_VLM_ERROR] {e}"
            
    except ImportError:
        print(f"      ⚠️  VLM client is not available; skipping critic evaluation")
        return True, ""
    except Exception as e:
        print(f"      ⚠️  Critic evaluation failed (setup error); skipping — NOT counted as a pass: {e}")
        import traceback
        traceback.print_exc()
        return True, f"[CRITIC_SKIPPED_VLM_ERROR] {e}"


def run_code_generation_test_only(
    feature: Dict[str, Any],
    state: AgentState,
    sample_ids: List[str],
    data_root: Path,
    find_image_paths_func,
    results_dir: Optional[Path] = None,
    conda_env: Optional[str] = None,
    max_cycles: Optional[int] = None,
    segmentation_mask_path: Optional[Path] = None,
    enable_critic: Optional[bool] = None
) -> Tuple[Optional[Path], CodeResult]:
    """Run code generation and test only the first sample (do not execute all samples)
    
    Args:
        feature: feature definition
        state: Agent state
        sample_ids: list of sample IDs (only used to get the first sample for testing)
        data_root: dataset root directory
        find_image_paths_func: function to find image paths
        results_dir: directory to save results (optional)
        conda_env: conda environment name (defaults to reading from config)
        max_cycles: maximum number of retries (defaults to reading from config)
        
    Returns:
        (extract_py_path, CodeResult) tuple; returns (None, CodeResult) on failure
    """
    from config import settings
    max_cycles = max_cycles or settings.code_max_retries
    generator = CodeGenerator()
    fixer = CodeFixer()
    executor = CodeExecutor(data_root, conda_env=conda_env)
    
    # Create the feature directory
    feature_name = feature.get("name", "unknown")
    if results_dir:
        feature_dir = results_dir / "features" / feature_name
        feature_dir.mkdir(parents=True, exist_ok=True)
    else:
        feature_dir = Path(f"/tmp/morphagent_features/{feature_name}")
        feature_dir.mkdir(parents=True, exist_ok=True)
    
    extract_py_path = feature_dir / "extract.py"
    guidance_message = ""
    code_result = CodeResult(code=None, prompt="", response="")
    
    print(f"\n[Code Generation] Generating and testing code for feature '{feature_name}' (testing only the first sample)")
    print(f"  Maximum retry count: {max_cycles}")
    
    # Before generating code, collect data statistics (from the first sample)
    data_statistics = {}
    planning_text = None
    
    if sample_ids:
        first_sample_id = sample_ids[0]
        first_sample_dir = data_root / first_sample_id
        
        # Get the full path information (image and segmentation)
        from tools.data_path_selector import get_data_path_selector
        selector = get_data_path_selector(verbose=False)
        path_result = selector.select_data_paths(
            first_sample_dir,
            {"method": "code"},
            state.get("research_summary", ""),
            method="code"
        )
        
        first_image_paths = []
        first_seg_paths = []
        if isinstance(path_result, dict):
            first_image_paths = path_result.get("image_paths", [])
            first_seg_paths = path_result.get("segmentation_paths", [])
        
        # Collect data statistics
        if first_image_paths:
            from tools.data_statistics import collect_data_statistics
            image_path = Path(first_image_paths[0])
            seg_paths = [Path(p) for p in first_seg_paths] if first_seg_paths else []
            
            print(f"  [Data Statistics] Collecting data statistics...")
            data_statistics = collect_data_statistics(
                image_path=image_path,
                segmentation_paths=seg_paths,
                dataset_description=state.get("research_summary", ""),
                sample_dir=first_sample_dir,
            )
            
            # If the state has no mask order information, try to read it from the segmentation summary file
            if not state.get("segmentation_mask_order") and data_root:
                try:
                    # Try to read segmentation_summary.json from the results directory
                    results_dir_parent = data_root.parent / "results" if "results" in str(data_root.parent) else None
                    if results_dir_parent and results_dir_parent.exists():
                        # Find the latest segmentation_summary.json
                        import glob
                        summary_files = sorted(glob.glob(str(results_dir_parent / "**/segmentation_summary.json")), reverse=True)
                        if summary_files:
                            with open(summary_files[0], 'r', encoding='utf-8') as f:
                                summary = json.load(f)
                                if summary.get("mask_order_description"):
                                    state["segmentation_mask_order"] = summary["mask_order_description"]
                                    print(f"  ✅ Read mask order information from the segmentation summary")
                except Exception as e:
                    print(f"  ⚠️  Failed to read the segmentation summary: {e}")
            
            # If still not available, use the one generated in data_statistics
            if not state.get("segmentation_mask_order") and data_statistics.get("segmentation_mask_order"):
                state["segmentation_mask_order"] = data_statistics["segmentation_mask_order"]
            
            state["data_statistics"] = data_statistics
            print(f"  ✅ Statistics collection completed")
            
            # Generate the CoT plan
            print(f"\n  [CoT Planning] Generating the code implementation plan...")
            planning_text = generator.generate_planning(
                feature=feature,
                state=state,
                data_statistics=data_statistics
            )
            if planning_text:
                print(f"  ✅ Plan generation completed")
                # Save the plan
                if results_dir:
                    planning_path = feature_dir / "code_planning.txt"
                    with open(planning_path, 'w', encoding='utf-8') as f:
                        f.write(planning_text)
                    print(f"  💾 Plan saved: {planning_path}")
            else:
                print(f"  ⚠️  Plan generation failed; code will be generated directly")
    
    for attempt in range(1, max_cycles + 1):
        print(f"\n  Attempt {attempt}/{max_cycles}")
        
        # Generate code (passing in the CoT plan)
        code_result = generator.generate(
            feature, 
            state, 
            guidance_message=guidance_message,
            planning_text=planning_text if attempt == 1 else None  # Only use the CoT plan on the first attempt
        )
        
        # Save the prompt and response
        if results_dir:
            prompt_path = feature_dir / (
                "code_prompt.json" if attempt == 1 else f"code_prompt_retry_{attempt - 1}.json"
            )
            with open(prompt_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "prompt": code_result.prompt,
                    "response": code_result.response,
                    "planning_response": code_result.planning_response if hasattr(code_result, 'planning_response') else None
                }, f, indent=2, ensure_ascii=False)
        
        if not code_result.code:
            print(f"    ❌ Unable to generate valid code")
            if attempt == max_cycles:
                return None, code_result
            continue
        
        # Save the code
        extract_py_path.write_text(code_result.code, encoding='utf-8')
        print(f"    ✅ Code saved: {extract_py_path}")
        
        # Test the first sample
        if sample_ids:
            first_sample_id = sample_ids[0]
            first_sample_dir = data_root / first_sample_id
            
            # Get the full path information for the first sample (including segmentation)
            from tools.data_path_selector import get_data_path_selector
            selector = get_data_path_selector(verbose=False)
            path_result = selector.select_data_paths(
                first_sample_dir,
                {"method": "code"},
                state.get("research_summary", ""),
                method="code"
            )
            
            first_image_paths = []
            first_seg_paths = []
            if isinstance(path_result, dict):
                first_image_paths = path_result.get("image_paths", [])
                first_seg_paths = path_result.get("segmentation_paths", [])
            else:
                # Backward compatibility with the old format
                first_image_paths = find_image_paths_func(first_sample_dir, "")
            
            if first_image_paths:
                first_image_path = Path(first_image_paths[0])
                # Convert segmentation paths to Path objects
                first_seg_paths_objs = [Path(p) for p in first_seg_paths] if first_seg_paths else []
                success, value, error_msg = executor.execute_single_sample(
                    extract_py_path, first_image_path, first_seg_paths_objs
                )
                
                if not success:
                    # Show the full error message for debugging
                    error_display = error_msg if len(error_msg) <= 1000 else error_msg[:1000] + "..."
                    print(f"    ❌ First sample test failed: {error_display}")
                    
                    # Check if it's a timeout error
                    if "timeout" in error_msg.lower() or "timed out" in error_msg:
                        print(f"    ⚠️  Timeout error detected")
                        if attempt == max_cycles:
                            return None, code_result
                        
                        # Generate fix plan
                        timeout_error = {"timeout": error_msg}
                        fix_plan = fixer.plan(feature, code_result.code, timeout_error)
                        if fix_plan:
                            _handle_code_fix(fix_plan, feature_dir, feature_name, conda_env=conda_env)
                            guidance_message = fix_plan.guidance_message
                        else:
                            guidance_message = "Code execution timed out. Please optimize the algorithm to reduce computation time, avoid nested loops over large arrays, and use vectorized operations."
                        continue
                    
                    # Other errors, generate fix plan
                    if attempt < max_cycles:
                        error_map = {first_sample_id: error_msg}
                        print(f"    🔧 Generating fix plan, error: {error_msg[:300]}...")
                        fix_plan = fixer.plan(feature, code_result.code, error_map)
                        if fix_plan:
                            print(f"    ✅ Fix plan generated")
                            _handle_code_fix(fix_plan, feature_dir, feature_name, conda_env=conda_env)
                            guidance_message = fix_plan.guidance_message
                        else:
                            print(f"    ⚠️  Failed to generate fix plan")
                            guidance_message = f"Previous error: {error_msg[:500]}"
                        continue
                    else:
                        # The last attempt failed
                        return None, code_result
                else:
                    print(f"    ✅ First sample test succeeded: {value}")
                    
                    # Check whether it is all zeros (existing logic)
                    if abs(value) < 1e-10:
                        print(f"    ⚠️  Detected a result of 0, which may indicate a problem")
                        if attempt < max_cycles:
                            zero_error = {
                                "all_samples": f"Test sample returned 0.0. This usually indicates a logic error in the code, such as: incorrect channel selection, wrong array indexing, incorrect data type handling, or a bug in the calculation formula."
                            }
                            fix_plan = fixer.plan(feature, code_result.code, zero_error)
                            if fix_plan:
                                _handle_code_fix(fix_plan, feature_dir, feature_name, conda_env=conda_env)
                                guidance_message = fix_plan.guidance_message
                            else:
                                guidance_message = "Test sample returned 0.0. Please check: (1) Are you using the correct channel/index? (2) Is the array indexing correct? (3) Is the calculation formula correct? (4) Are there any data type issues?"
                            continue
                    
                    # Call the critic agent to evaluate result reasonableness
                    print(f"    [Critic Agent] Evaluating result reasonableness...")
                    critic_passed, critic_feedback = evaluate_result_with_critic(
                        feature=feature,
                        image_path=first_image_path,
                        result_value=value,
                        code=code_result.code,
                        results_dir=results_dir,
                        feature_dir=feature_dir,
                        enable_critic=enable_critic  # Pass the critic agent enabled state
                    )
                    
                    if not critic_passed:
                        print(f"    ❌ Critic Agent evaluation not passed: {critic_feedback[:200]}...")
                        if attempt < max_cycles:
                            # Add the critic feedback to guidance_message
                            if guidance_message:
                                guidance_message = f"{guidance_message}\n\n[Critic Agent Feedback] {critic_feedback}"
                            else:
                                guidance_message = f"[Critic Agent Feedback] The result was evaluated as unreasonable. {critic_feedback}"
                            continue
                        else:
                            # Last attempt; accept this code even if the critic does not pass (at least the code runs)
                            print(f"    ⚠️  Reached maximum retry count; accepting this code even though the critic did not pass")
                            return extract_py_path, code_result

                    # Test passed; only claim critic passed when it actually evaluated (not skipped on VLM error)
                    if isinstance(critic_feedback, str) and critic_feedback.startswith("[CRITIC_SKIPPED_VLM_ERROR]"):
                        print(f"    ✅ Test passed (critic skipped due to VLM error)")
                    else:
                        print(f"    ✅ Test passed and critic evaluation passed")
                    return extract_py_path, code_result
        
        # If there are no samples, also return the code path (although it cannot be tested)
        return extract_py_path, code_result
    
    # All attempts failed
    print(f"    ❌ Reached maximum retry count; giving up on this feature")
    return None, code_result


def merge_feature_codes(
    features: List[Dict[str, Any]],
    code_paths: List[Path],
    state: AgentState,
    results_dir: Optional[Path] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Use an LLM to merge multiple feature codes into a single unified code
    
    Args:
        features: list of feature definitions
        code_paths: list of corresponding code file paths
        state: Agent state
        results_dir: directory to save results (optional)
        
    Returns:
        (merged_code, prompt_response) tuple; returns (None, None) on failure
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from config import settings
    
    print(f"\n[Code Merger] Merging code for {len(features)} features...")
    
    # Read all the code
    feature_codes = []
    for i, (feature, code_path) in enumerate(zip(features, code_paths)):
        if not code_path or not code_path.exists():
            print(f"  ⚠️  Warning: the code file for feature {i+1} ({feature.get('name', 'unknown')}) does not exist: {code_path}")
            continue
        
        code_content = code_path.read_text(encoding='utf-8')
        feature_codes.append({
            "index": i + 1,
            "name": feature.get("name", f"feature_{i+1}"),
            "description": feature.get("description", ""),
            "code": code_content
        })
        print(f"  ✅ Read feature {i+1}: {feature.get('name', 'unknown')}")
    
    if not feature_codes:
        print(f"  ❌ There is no valid code to merge")
        return None, None
    
    # Build the merge prompt: explicitly require extract_all(img, seg) with seg as a key-value dict, to avoid the LLM generating multiple parameters or truncating
    system_prompt = """You are an expert Python programmer specializing in image feature extraction.

Your task is to merge multiple feature extraction functions into a single unified function that processes all features in one pass.

====================
CRITICAL – Function signature (MUST follow exactly)
====================
- The merged function MUST be:  def extract_all(img, seg):
- Exactly TWO parameters:  img  (numpy array, the image),  seg  (a single dict).
- seg is a dictionary (key-value): keys = filename stems of mask files (e.g. "mask_cell", "mask_nucleus", "mask_bundle", "mask_filament", "mask_droplet"). Values = numpy arrays.
- The runner that calls extract_all builds seg like this:  seg["mask_cell"] = array,  seg["mask_nucleus"] = array,  etc. So you MUST access masks only by key:  seg.get("mask_cell"),  seg.get("mask_nucleus"),  etc. Do NOT use positional args (no extract_all(img, mask1, mask2, ...)).
- Inside the function, you may assign  cell_mask = seg.get("mask_cell"),  nuc_mask = seg.get("mask_nucleus"),  etc. to avoid name conflicts between features. There is only one seg dict.

====================
Other requirements
====================
1. Return a single dict:  {"feature_name_1": value1, "feature_name_2": value2, ...}. Use the exact feature names as in the original functions.
2. Process all features in the order provided. Wrap each feature block in try/except so one failure does not break others; on exception set that feature to float('nan') or 0.0 as appropriate.
3. Every try MUST have an except or finally block. Every if/for/while MUST have a non-empty body. Do NOT truncate the code – output complete, syntactically valid Python to the end of the function (including return results).
4. Import all required libraries at the top of the file. Preserve the original logic and calculations from each feature.
5. CRITICAL: You MUST output the entire function to the very end: the last feature block must have its except/finally, and the function MUST end with a line like \"return results\". Do not stop early; the response must be complete and runnable.

Return ONLY the Python code, no markdown, no explanations."""

    # Build the user prompt
    features_info = []
    for fc in feature_codes:
        features_info.append(f"""
Feature {fc['index']}: {fc['name']}
Description: {fc['description']}
Code:
```python
{fc['code']}
```
""")
    
    user_prompt = f"""Merge these {len(feature_codes)} feature extraction functions into ONE function:

  def extract_all(img, seg):

where seg is a single dict: keys = mask file stems (e.g. "mask_cell", "mask_nucleus"). Access masks only via seg.get("mask_cell"), etc. Return a dict of feature_name -> value. Use try/except per feature so one failure does not break others. Do not truncate – output the complete function including final return.

{''.join(features_info)}

Return ONLY the Python code, starting with imports and ending with the full extract_all function. You MUST include the final \"return results\" and every try must have its except block – do not truncate."""

    # Call the LLM: the merged code is fairly long, so use a larger max_tokens to reduce the chance of truncation
    merge_max_tokens = getattr(settings, "merge_max_tokens", None)
    max_tokens = merge_max_tokens if merge_max_tokens is not None else max(settings.llm_max_tokens, 100000)
    from config import make_chat_llm

    llm = make_chat_llm(
        temperature=0,
        max_tokens=max_tokens,
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    
    print(f"  [LLM] Generating merged code...")
    try:
        response = llm.invoke(messages)
        merged_code = response.content if hasattr(response, 'content') else str(response)
        
        # Extract the code (it may contain a markdown code block)
        merged_code = merged_code.strip()
        if merged_code.startswith('```'):
            # Remove the markdown code block
            lines = merged_code.split('\n')
            merged_code = '\n'.join([l for l in lines if not l.strip().startswith('```')])
            # Remove the ```python or ``` from the first and last lines
            if lines[0].strip().startswith('```'):
                merged_code = '\n'.join(lines[1:])
            if merged_code.strip().endswith('```'):
                merged_code = merged_code.rsplit('```', 1)[0].strip()
        
        # Syntax check after merging, to avoid writing code that cannot compile and causes full execution to fail
        def _compile_check(code: str) -> bool:
            try:
                compile(code, "<merged_extract_all>", "exec")
                return True
            except SyntaxError:
                return False

        syntax_ok = _compile_check(merged_code)
        first_error = None
        if not syntax_ok:
            try:
                compile(merged_code, "<merged_extract_all>", "exec")
            except SyntaxError as e:
                first_error = e
            # One retry: ask the LLM to complete the missing except/return
            if first_error is not None:
                print(f"  ⚠️  Merged code has a syntax error; attempting one completion retry: {first_error}")
                fix_prompt = f"""The following Python code has a SyntaxError: {first_error}

The code is likely truncated. Please output the COMPLETE corrected code: add any missing except or finally blocks for every try, and ensure the function ends with \"return results\". Output the full file from the first line (imports) to the last line (return results).

Code that needs to be completed/fixed:

```python
{merged_code}
```

Return ONLY the complete, corrected Python code (full file), no markdown wrapper."""
                try:
                    fix_response = llm.invoke([HumanMessage(content=fix_prompt)])
                    fixed_code = (fix_response.content if hasattr(fix_response, "content") else str(fix_response)).strip()
                    if fixed_code.startswith("```"):
                        lines = fixed_code.split("\n")
                        if lines[0].strip().startswith("```"):
                            fixed_code = "\n".join(lines[1:])
                        if fixed_code.strip().endswith("```"):
                            fixed_code = fixed_code.rsplit("```", 1)[0].strip()
                    if _compile_check(fixed_code):
                        merged_code = fixed_code
                        syntax_ok = True
                except Exception as retry_e:
                    print(f"  ⚠️  Completion retry failed: {retry_e}")
            if not syntax_ok:
                err_msg = first_error if first_error is not None else "SyntaxError (unknown)"
                print(f"  ❌ Merged code syntax error (SyntaxError/IndentationError): {err_msg}")
                if results_dir:
                    err_path = results_dir / "merged_feature_code_syntax_error.txt"
                    err_path.write_text(
                        f"SyntaxError: {err_msg}\n\nCode preview (first 2000 chars):\n{merged_code[:2000]}",
                        encoding="utf-8"
                    )
                    print(f"  💾 Error message and code preview saved: {err_path}")
                return None, None
        
        # Extra completeness check: ensure every feature has an explicit assignment in the merged code, rather than only relying on a shared default dict
        # Check pattern: whether there is an assignment of the form results[\"feature_name\"] or results['feature_name']
        missing_feature_assignments: List[str] = []
        for fc in feature_codes:
            fname = fc.get("name", "")
            if not fname:
                continue
            pattern_dq = f'results["{fname}"]'
            pattern_sq = f"results['{fname}']"
            if pattern_dq not in merged_code and pattern_sq not in merged_code:
                missing_feature_assignments.append(fname)
        
        if missing_feature_assignments:
            # Record the incomplete merged code and trigger a fallback to per-feature execution mode
            print(
                "  ❌ The merged code has no explicit assignment for the following features (it may only use the default dict); "
                "to avoid producing large batches of all-0/NaN, this merge will be abandoned and fall back to per-feature execution mode:"
            )
            print(f"     {', '.join(missing_feature_assignments)}")
            if results_dir:
                incomplete_path = results_dir / "merged_feature_code_incomplete.py"
                incomplete_path.write_text(merged_code, encoding="utf-8")
                summary_path = results_dir / "merged_feature_code_incomplete.txt"
                summary_path.write_text(
                    "No explicit results['name'] assignment was detected in the merged code for the following features; "
                    "the merge result is considered incomplete and triggers fallback mode:\n"
                    + "\n".join(missing_feature_assignments),
                    encoding="utf-8",
                )
                print(f"  💾 Incomplete merged code and summary saved: {incomplete_path}, {summary_path}")
            return None, None

        print(f"  ✅ Merged code generated successfully and passed syntax and completeness checks ({len(merged_code)} characters)")
        
        # Save the merged code
        if results_dir:
            merged_code_path = results_dir / "merged_feature_code.py"
            merged_code_path.write_text(merged_code, encoding='utf-8')
            print(f"  💾 Merged code saved: {merged_code_path}")
        
        return merged_code, response.content if hasattr(response, 'content') else str(response)
        
    except Exception as e:
        print(f"  ❌ Merged code generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def execute_merged_code(
    merged_code: str,
    feature_names: List[str],
    sample_ids: List[str],
    data_root: Path,
    find_image_paths_func,
    results_dir: Optional[Path] = None,
    conda_env: Optional[str] = None,
    segmentation_mask_path: Optional[Path] = None,
    num_workers: int = 1
) -> ExtractionResult:
    """Execute the merged code, processing all samples and extracting all features
    
    Args:
        merged_code: the merged code string
        feature_names: list of feature names (in order)
        sample_ids: list of sample IDs
        data_root: dataset root directory
        find_image_paths_func: function to find image paths
        results_dir: directory to save results (optional)
        conda_env: conda environment name (defaults to reading from config)
        num_workers: number of parallel processes (default 1, i.e. serial processing)
        
    Returns:
        ExtractionResult object containing all feature values for all samples (in the original order)
    """
    from config import settings
    from tqdm import tqdm
    
    if num_workers <= 1:
        # Serial processing (original logic)
        print(f"\n[Code Execution] Executing merged code (serial mode), processing {len(sample_ids)} samples, extracting {len(feature_names)} features...")
        return _execute_merged_code_serial(
            merged_code, feature_names, sample_ids, data_root, 
            find_image_paths_func, results_dir, conda_env, segmentation_mask_path
        )
    else:
        # Parallel processing
        print(f"\n[Code Execution] Executing merged code (parallel mode, {num_workers} processes), processing {len(sample_ids)} samples, extracting {len(feature_names)} features...")
        return _execute_merged_code_parallel(
            merged_code, feature_names, sample_ids, data_root,
            find_image_paths_func, results_dir, conda_env, segmentation_mask_path, num_workers
        )


def _execute_merged_code_serial(
    merged_code: str,
    feature_names: List[str],
    sample_ids: List[str],
    data_root: Path,
    find_image_paths_func,
    results_dir: Optional[Path] = None,
    conda_env: Optional[str] = None,
    segmentation_mask_path: Optional[Path] = None
) -> ExtractionResult:
    """Serially execute the merged code (original logic)"""
    from config import settings
    from tqdm import tqdm
    
    print(f"\n[Code Execution] Executing merged code, processing {len(sample_ids)} samples, extracting {len(feature_names)} features...")
    
    # Create a temporary extract.py file
    if results_dir:
        merged_dir = results_dir / "merged_features"
        merged_dir.mkdir(parents=True, exist_ok=True)
        extract_py_path = merged_dir / "extract_all.py"
    else:
        merged_dir = Path("/tmp/morphagent_merged_features")
        merged_dir.mkdir(parents=True, exist_ok=True)
        extract_py_path = merged_dir / "extract_all.py"
    
    extract_py_path.write_text(merged_code, encoding='utf-8')
    print(f"  ✅ Merged code saved: {extract_py_path}")
    
    # Create the executor
    executor = CodeExecutor(data_root, conda_env=conda_env)
    
    # Create the result dict: {sample_id: {feature_name: value}}
    all_values = {}
    all_errors = {}
    
    # Create the log file
    log_file = None
    if results_dir:
        log_file = merged_dir / "execution_log.txt"
        log_fp = open(log_file, 'w', encoding='utf-8')
    else:
        log_fp = None
    
    try:
        # Process all samples
        for sample_id in tqdm(sample_ids, desc="  Executing merged code", leave=False, ncols=80):
            sample_dir = data_root / sample_id
            if not sample_dir.exists():
                error_msg = f"Sample directory does not exist: {sample_dir}"
                all_errors[sample_id] = error_msg
                if log_fp:
                    log_fp.write(f"{sample_id}: ❌ {error_msg}\n")
                continue
            
            # Get the image path and segmentation paths
            from tools.data_path_selector import get_data_path_selector
            selector = get_data_path_selector(verbose=False)
            path_result = selector.select_data_paths(
                sample_dir,
                {"method": "code"},
                None,
                method="code"
            )
            
            image_paths = []
            seg_paths = []
            if isinstance(path_result, dict):
                image_paths = path_result.get("image_paths", [])
                seg_paths = path_result.get("segmentation_paths", [])
            else:
                # Backward compatibility with the old format
                image_paths = find_image_paths_func(sample_dir, "")
            
            if not image_paths:
                error_msg = f"No image file found for sample {sample_id}"
                all_errors[sample_id] = error_msg
                if log_fp:
                    log_fp.write(f"{sample_id}: ❌ {error_msg}\n")
                continue
            
            # Execute the code
            image_path = Path(image_paths[0])
            seg_paths_objs = [Path(p) for p in seg_paths] if seg_paths else []
            
            success, result_value, error_msg = executor.execute_single_sample(
                extract_py_path, image_path, seg_paths_objs
            )
            
            if success:
                # Parse the result (should be in dict format)
                if isinstance(result_value, dict):
                    # The result is already in dict format
                    all_values[sample_id] = result_value
                else:
                    # If the result is not a dict, try to parse JSON
                    try:
                        import json
                        if isinstance(result_value, str):
                            result_dict = json.loads(result_value)
                        else:
                            result_dict = {"unknown": result_value}
                        all_values[sample_id] = result_dict
                    except:
                        # If parsing fails, assume there is only one feature
                        all_values[sample_id] = {feature_names[0]: result_value} if feature_names else {"unknown": result_value}
                
                if log_fp:
                    log_fp.write(f"{sample_id}: ✅ {all_values[sample_id]}\n")
            else:
                all_errors[sample_id] = error_msg or "Unknown error"
                if log_fp:
                    log_fp.write(f"{sample_id}: ❌ {error_msg}\n")
                    log_fp.flush()
    
    finally:
        if log_fp:
            log_fp.close()
            if log_file:
                print(f"  ✅ Detailed results saved to: {log_file}")
    
    # Summarize the results
    total_samples = len(sample_ids)
    successful_samples = len(all_values)
    print(f"  ✅ Execution completed: {successful_samples}/{total_samples} samples succeeded")
    
    # Return a dict containing all feature values
    return ExtractionResult(
        values=all_values,  # Format: {sample_id: {feature_name: value}}
        errors=all_errors
    )


def _process_sample_chunk_worker(args):
    """Worker function that processes a batch of samples (module level, picklable)"""
    chunk_with_indices, data_root_str, extract_py_path_str, conda_env, feature_names_list = args
    
    from pathlib import Path
    from tools.code_executor import CodeExecutor
    from tools.data_path_selector import get_data_path_selector
    
    data_root = Path(data_root_str)
    extract_py_path = Path(extract_py_path_str)
    
    # Create the executor
    executor = CodeExecutor(data_root, conda_env=conda_env)
    
    # Process all samples in this batch
    chunk_results = {}  # {original_index: (sample_id, result_dict or error_msg, is_success)}
    
    for original_index, sample_id in chunk_with_indices:
        sample_dir = data_root / sample_id
        if not sample_dir.exists():
            chunk_results[original_index] = (sample_id, f"Sample directory does not exist: {sample_dir}", False)
            continue
        
        # Get the image path and segmentation paths
        selector = get_data_path_selector(verbose=False)
        path_result = selector.select_data_paths(
            sample_dir,
            {"method": "code"},
            None,
            method="code"
        )
        
        image_paths = []
        seg_paths = []
        if isinstance(path_result, dict):
            image_paths = path_result.get("image_paths", [])
            seg_paths = path_result.get("segmentation_paths", [])
        else:
            # Backward compatibility with the old format
            image_paths = path_result if isinstance(path_result, list) else []
        
        if not image_paths:
            chunk_results[original_index] = (sample_id, f"No image file found for sample {sample_id}", False)
            continue
        
        # Execute the code
        image_path = Path(image_paths[0])
        seg_paths_objs = [Path(p) for p in seg_paths] if seg_paths else []
        
        success, result_value, error_msg = executor.execute_single_sample(
            extract_py_path, image_path, seg_paths_objs
        )
        
        if success:
            # Parse the result (should be in dict format)
            if isinstance(result_value, dict):
                result_dict = result_value
            else:
                # If the result is not a dict, try to parse JSON
                try:
                    import json
                    if isinstance(result_value, str):
                        result_dict = json.loads(result_value)
                    else:
                        result_dict = {"unknown": result_value}
                except:
                    # If parsing fails, assume there is only one feature
                    result_dict = {feature_names_list[0]: result_value} if feature_names_list else {"unknown": result_value}
            
            chunk_results[original_index] = (sample_id, result_dict, True)
        else:
            chunk_results[original_index] = (sample_id, error_msg or "Unknown error", False)
    
    return chunk_results


def _execute_merged_code_parallel(
    merged_code: str,
    feature_names: List[str],
    sample_ids: List[str],
    data_root: Path,
    find_image_paths_func,
    results_dir: Optional[Path] = None,
    conda_env: Optional[str] = None,
    segmentation_mask_path: Optional[Path] = None,
    num_workers: int = 8
) -> ExtractionResult:
    """Execute the merged code in parallel (multi-process processing)
    
    Key point: preserve the result order, ensuring it matches the order of the input sample_ids
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    import multiprocessing
    
    # Create a temporary extract.py file (shared by all processes)
    if results_dir:
        merged_dir = results_dir / "merged_features"
        merged_dir.mkdir(parents=True, exist_ok=True)
        extract_py_path = merged_dir / "extract_all.py"
    else:
        merged_dir = Path("/tmp/morphagent_merged_features")
        merged_dir.mkdir(parents=True, exist_ok=True)
        extract_py_path = merged_dir / "extract_all.py"
    
    extract_py_path.write_text(merged_code, encoding='utf-8')
    print(f"  ✅ Merged code saved: {extract_py_path}")
    
    # Pre-split the list of sample IDs, preserving the order
    def split_samples(sample_ids: List[str], num_workers: int) -> List[List[Tuple[int, str]]]:
        """Split the list of sample IDs into num_workers parts, each containing (original_index, sample_id) tuples to preserve order"""
        chunks = []
        chunk_size = (len(sample_ids) + num_workers - 1) // num_workers  # Round up
        
        for i in range(0, len(sample_ids), chunk_size):
            # Use the original index to ensure the order is correct
            chunk = [(i + j, sample_ids[i + j]) for j in range(len(sample_ids[i:i+chunk_size]))]
            chunks.append(chunk)
        
        return chunks
    
    # Split the samples
    sample_chunks = split_samples(sample_ids, num_workers)
    print(f"  📊 Samples split into {len(sample_chunks)} batches (out of {len(sample_ids)} samples)")
    
    # Prepare the arguments (Path needs to be converted to a string so it can be pickled)
    chunk_args = [
        (chunk, str(data_root), str(extract_py_path), conda_env, feature_names)
        for chunk in sample_chunks
    ]
    
    # Use a process pool for parallel processing
    all_values = {}  # {sample_id: {feature_name: value}}
    all_errors = {}  # {sample_id: error_msg}
    
    # Create the log file
    log_file = None
    if results_dir:
        log_file = merged_dir / "execution_log.txt"
        log_fp = open(log_file, 'w', encoding='utf-8')
    else:
        log_fp = None
    
    try:
        # Use ProcessPoolExecutor for parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks (using the module-level function)
            future_to_chunk = {
                executor.submit(_process_sample_chunk_worker, args): i 
                for i, args in enumerate(chunk_args)
            }
            
            # Use tqdm to display progress
            with tqdm(total=len(sample_ids), desc="  Executing merged code in parallel", ncols=80) as pbar:
                # Collect results in completion order (but preserve the original index order)
                completed_results = {}  # {chunk_index: chunk_results}
                
                for future in as_completed(future_to_chunk):
                    chunk_index = future_to_chunk[future]
                    try:
                        chunk_results = future.result()
                        completed_results[chunk_index] = chunk_results
                        
                        # Update the progress bar
                        pbar.update(len(chunk_results))
                    except Exception as e:
                        print(f"  ⚠️  Batch {chunk_index} processing failed: {e}")
                        import traceback
                        traceback.print_exc()
                        # Record all samples in this batch as errors
                        chunk = sample_chunks[chunk_index]
                        for original_index, sample_id in chunk:
                            all_errors[sample_id] = f"Batch processing failed: {e}"
                            if log_fp:
                                log_fp.write(f"{sample_id}: ❌ Batch processing failed: {e}\n")
                
                # Merge results in the original order (key point: preserve order)
                for chunk_index in sorted(completed_results.keys()):
                    chunk_results = completed_results[chunk_index]
                    # Sort by original_index to ensure order
                    for original_index in sorted(chunk_results.keys()):
                        sample_id, result, is_success = chunk_results[original_index]
                        if is_success:
                            # result should be a dict {feature_name: value}
                            if isinstance(result, dict):
                                all_values[sample_id] = result
                                if log_fp:
                                    log_fp.write(f"{sample_id}: ✅ {result}\n")
                                    log_fp.flush()
                            else:
                                # If it is not a dict, record it as an error
                                all_errors[sample_id] = f"Result format error: expected a dict, got {type(result)}"
                                if log_fp:
                                    log_fp.write(f"{sample_id}: ❌ Result format error: {type(result)}\n")
                                    log_fp.flush()
                        else:
                            all_errors[sample_id] = result
                            if log_fp:
                                log_fp.write(f"{sample_id}: ❌ {result}\n")
                                log_fp.flush()
    
    finally:
        if log_fp:
            log_fp.close()
            if log_file:
                print(f"  ✅ Detailed results saved to: {log_file}")
    
    # Summarize the results
    total_samples = len(sample_ids)
    successful_samples = len(all_values)
    print(f"  ✅ Execution completed: {successful_samples}/{total_samples} samples succeeded")
    
    # Return the results (already in the original order)
    return ExtractionResult(
        values=all_values,  # Format: {sample_id: {feature_name: value}}
        errors=all_errors
    )
