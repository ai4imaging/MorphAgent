import sys
import json
import traceback
from pathlib import Path
import numpy as np
import os

# Note: We only pre-import numpy as a minimal convenience.
# All other packages (scipy, skimage, etc.) should be imported inside the extract() function.
# This ensures the code is self-contained and can auto-install missing packages.

# Import image loading libraries
try:
    import tifffile
except ImportError:
    tifffile = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import imageio
except ImportError:
    imageio = None

try:
    import cv2
except ImportError:
    cv2 = None

def json_default(value):
    """Convert NumPy values returned by generated features to JSON values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')

# Note: skimage, scipy, and other packages should be imported inside extract() function
# We don't pre-import them here to ensure code is self-contained
def load_image(image_path):
    """Load image from file, handling different formats (TIFF, PNG, JPG, etc.)"""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f'Image not found: {{image_path}}')
    
    file_ext = image_path.suffix.lower()
    
    # TIFF files: use tifffile (primary method)
    if file_ext in ['.tif', '.tiff']:
        if tifffile is None:
            raise ImportError('tifffile is required for TIFF files')
        try:
            img = tifffile.imread(str(image_path))
            return np.asarray(img)
        except Exception as e:
            # If tifffile fails (e.g., not a real TIFF), try PIL as fallback
            if Image is not None:
                try:
                    img = Image.open(str(image_path))
                    return np.array(img)
                except:
                    pass
            raise ValueError(f'Failed to load TIFF file: {{e}}')
    
    # PNG, JPG, etc.: use PIL (preferred) or imageio
    if Image is not None:
        try:
            img = Image.open(str(image_path))
            arr = np.array(img)
            # Convert RGBA to RGB if needed
            if len(arr.shape) == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            return arr
        except Exception as e:
            if imageio is None:
                raise ValueError(f'Failed to load image with PIL: {{e}}')
    
    if imageio is not None:
        try:
            return imageio.imread(str(image_path))
        except Exception as e:
            raise ValueError(f'Failed to load image with imageio: {{e}}')
    
    raise ImportError('No image loading library available (PIL, imageio, or tifffile)')

try:
    # Get image path from command line argument
    if len(sys.argv) < 2:
        raise ValueError('Image path required as command line argument')
    image_path_str = sys.argv[1]
    image_path = Path(image_path_str).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f'Image not found: {{image_path}}')
    
    arr = load_image(image_path)

    # Load segmentation as key-value dict (key = filename stem, e.g. mask_cell, mask_nucleus)
    seg = dict()  # seg['mask_cell'], seg['mask_nucleus'], etc. Keys from filenames.
    if len(sys.argv) >= 3 and tifffile is not None:
        for i in range(2, len(sys.argv)):
            seg_path = Path(sys.argv[i]).resolve()
            if seg_path.exists():
                key = seg_path.stem  # e.g. mask_cell from mask_cell.tif
                seg[key] = tifffile.imread(str(seg_path))
                print(f'Seg loaded {key!r} from: {seg_path.name}', file=sys.stderr)
            else:
                print(f'Warning: Segmentation file not found: {seg_path}', file=sys.stderr)
    # Auto-detect from sample_dir/segmentation/ if no paths passed
    if len(seg) == 0:
        image_dir = image_path.parent
        seg_dir = image_dir / 'segmentation'
        if seg_dir.exists() and tifffile is not None:
            for seg_file in sorted(seg_dir.glob('*.tif')):
                key = seg_file.stem
                seg[key] = tifffile.imread(str(seg_file))
                print(f'Auto-loaded seg {key!r} from: {seg_file.name}', file=sys.stderr)
    # Backward compat: list in sorted key order for extract(img, *segmentation_masks)
    segmentation_masks = [seg[k] for k in sorted(seg.keys())] if seg else []

    # Handle different image formats
    # Note: We preserve the original shape for code features
    # The LLM-generated code should handle the data format based on dataset description
    # Only transpose if it's clearly a z-stack (many slices, not channels)
    if arr.ndim == 3:
        # If first dimension is small (< 20), it's likely channels, keep as is
        # If first dimension is large, it might be z-stack, but let code decide
        # For now, preserve original format to match code expectations
        pass  # Keep original shape - code will handle format based on dataset description

    prepared = arr

    # Import extract function
    extract_py_dir = Path('<BUNDLE_ROOT>/MorphAgent/demo/data/results/completed_demo_run/round_1/features/distal_to_proximal_axon_tau_intensity_ratio')
    if str(extract_py_dir) not in sys.path:
        sys.path.insert(0, str(extract_py_dir))

    extract_py_path = Path('<BUNDLE_ROOT>/MorphAgent/demo/data/results/completed_demo_run/round_1/features/distal_to_proximal_axon_tau_intensity_ratio/extract.py')
    if not extract_py_path.exists():
        raise FileNotFoundError(f'extract.py not found at {extract_py_path}')

    # Read and execute extract.py
    with open(extract_py_path, 'r', encoding='utf-8') as f:
        extract_code = f.read()

    # Execute extract.py - code should import all required packages inside the function
    # We only provide numpy as a minimal convenience, everything else should be imported in extract()
    exec_globals = {
        'np': np,  # Only numpy is pre-imported as a minimal convenience
    }

    # Function to auto-install missing packages
    def auto_install_package(package_name, conda_env=None):
        """Try to install a missing package"""
        import subprocess
        import sys
        try:
            if conda_env:
                cmd = ['conda', 'run', '-n', conda_env, 'pip', 'install', package_name]
            else:
                cmd = [sys.executable, '-m', 'pip', 'install', package_name]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                return True
            return False
        except Exception:
            return False

    # Get conda environment for auto-install
    conda_env_for_install = 'morphagent'
    if conda_env_for_install is None:
        conda_env_for_install = os.environ.get('CONDA_ENV', 'morphagent')

    # Try to execute the code, with automatic package installation on ImportError
    max_import_retries = 2
    for retry in range(max_import_retries):
        try:
            code_obj = compile(extract_code, str(extract_py_path), 'exec')
            exec(code_obj, exec_globals)
            break  # Success, exit retry loop
        except ImportError as e:
            import_error_str = str(e)
            # Extract package name from ImportError message
            # Examples: "No module named 'scipy.stats'" -> "scipy"
            #          "cannot import name 'stats' from 'scipy'" -> "scipy"
            package_name = None
            if "No module named" in import_error_str:
                # Extract module name (e.g., "scipy.stats" -> "scipy")
                import re
                match = re.search(r"No module named ['\"]([^'\"]+)['\"]", import_error_str)
                if match:
                    full_module = match.group(1)
                    # Get top-level package (e.g., "scipy.stats" -> "scipy")
                    package_name = full_module.split('.')[0]
            elif "cannot import name" in import_error_str:
                # Extract module name (e.g., "cannot import name 'stats' from 'scipy'" -> "scipy")
                import re
                match = re.search(r"from ['\"]([^'\"]+)['\"]", import_error_str)
                if match:
                    package_name = match.group(1).split('.')[0]

            if package_name and retry < max_import_retries - 1:
                # Try to install the package
                print(f'[Auto-install] Attempting to install missing package: {package_name}', file=sys.stderr)
                if auto_install_package(package_name, conda_env=conda_env_for_install):
                    print(f'[Auto-install] Successfully installed {package_name}, retrying execution...', file=sys.stderr)
                    continue  # Retry execution
                else:
                    print(f'[Auto-install] Failed to install {package_name}', file=sys.stderr)

            # If we can't install or this is the last retry, raise the error
            exec_error = f'Import error in extract.py: {e}. Tried to auto-install but failed.'
            exec_error_tb = traceback.format_exc()
            raise ValueError(f'{exec_error}\nTraceback:\n{exec_error_tb}')
        except SyntaxError as e:
            exec_error = f'Syntax error in extract.py: {e}'
            exec_error_tb = traceback.format_exc()
            raise ValueError(f'{exec_error}\nTraceback:\n{exec_error_tb}')
        except Exception as e:
            exec_error = f'Error executing extract.py: {type(e).__name__}: {e}'
            exec_error_tb = traceback.format_exc()
            raise ValueError(f'{exec_error}\nTraceback:\n{exec_error_tb}')

    # Get extract function (try extract_all first for merged code, then extract)
    extract_func = exec_globals.get('extract_all', None)
    if extract_func is None:
        extract_func = exec_globals.get('extract', None)
    if extract_func is None:
        available_names = [k for k in exec_globals.keys() if not k.startswith('_')]
        error_msg = f'extract or extract_all function not found in extract.py. Available names: {available_names}'
        raise ValueError(error_msg)

    # Execute extraction
    # Try to call with appropriate arguments based on function signature
    import inspect
    sig = inspect.signature(extract_func)
    param_names = list(sig.parameters.keys())
    num_params = len(param_names)
    
    # Determine which arguments to pass based on function signature
    # Preferred: extract(img, seg) with seg = dict of name -> array (keys from filenames)
    if num_params == 1:
        raw = extract_func(prepared)
    elif num_params == 2:
        second = param_names[1]
        if second in ('seg', 'segmentation') and isinstance(seg, dict):
            raw = extract_func(prepared, seg)
        elif len(segmentation_masks) > 0:
            raw = extract_func(prepared, segmentation_masks[0])
        else:
            raw = extract_func(prepared)
    else:
        if param_names[1] in ('seg', 'segmentation') and isinstance(seg, dict):
            raw = extract_func(prepared, seg)
        elif 'segmentation_masks' in param_names:
            raw = extract_func(prepared, segmentation_masks)
        elif len(param_names) == 2 + len(segmentation_masks):
            args = [prepared] + segmentation_masks
            raw = extract_func(*args)
        else:
            num_mask_params = num_params - 1
            mask_args = segmentation_masks[:num_mask_params] if len(segmentation_masks) >= num_mask_params else segmentation_masks
            while len(mask_args) < num_mask_params:
                mask_args.append(None)
            raw = extract_func(prepared, *mask_args)

    # Handle return value: could be float, dict, or other types
    if isinstance(raw, dict):
        # If function returns a dict (merged features), use it directly
        result = raw
    elif isinstance(raw, (int, float)):
        # If function returns a single value, convert to float
        result = float(raw)
    else:
        # Try to convert to float, or use as-is
        try:
            result = float(raw)
        except (ValueError, TypeError):
            result = raw

    # Output result as JSON
    print(json.dumps({'success': True, 'value': result}, default=json_default))
    sys.exit(0)

except Exception as e:
    error_msg = traceback.format_exc()
    print(json.dumps({'success': False, 'error': str(e), 'traceback': error_msg}))
    sys.exit(1)