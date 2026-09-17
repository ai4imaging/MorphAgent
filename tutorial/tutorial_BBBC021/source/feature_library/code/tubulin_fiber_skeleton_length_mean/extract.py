def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.morphology import skeletonize

    # 1. Input Validation and Preparation
    if img is None:
        return 0.0
    
    # Handle dimensionality according to dataset format (512, 512, 3)
    # Channel 1 is Tubulin (Green)
    if img.ndim == 3 and img.shape[2] >= 2:
        tubulin = img[..., 1]
    elif img.ndim == 2:
        # Fallback for potential single-channel inputs
        tubulin = img
    else:
        return 0.0

    # Convert to float32 for processing
    tubulin = tubulin.astype(np.float32)

    # Intensity normalization [0, 1]
    # Using percentile to be robust against hot pixels
    min_val = np.min(tubulin)
    max_val = np.percentile(tubulin, 99.5)
    
    if max_val > min_val:
        tubulin = (tubulin - min_val) / (max_val - min_val)
    elif max_val > 0:
        tubulin = tubulin / max_val
    tubulin = np.clip(tubulin, 0.0, 1.0)

    # 2. Segmentation Mask Selection
    # Filter out any None values passed in masks
    valid_masks = [m for m in segmentation_masks if m is not None]
    
    if not valid_masks:
        # Without cell boundaries, we cannot compute "mean per cell"
        return 0.0
    
    # Heuristic: The last mask is typically the cell mask (e.g. [nuclei, cells])
    # If only one mask is present, use it.
    cell_mask = valid_masks[-1]
    
    # Ensure mask is integer for labeling
    cell_mask = cell_mask.astype(np.int32)

    # 3. Feature Extraction: Tubulin Skeleton
    
    # Pre-processing: Gaussian blur to smooth noise before thresholding
    # Sigma=1.0 is appropriate for 512x512 microscopy images to reduce noise-induced spurs
    blurred = ndimage.gaussian_filter(tubulin, sigma=1.0)
    
    # Binarization
    # Use Otsu's method to find a global threshold for the tubulin structure
    try:
        thresh = threshold_otsu(blurred)
        binary_tubulin = blurred > thresh
    except Exception:
        # Fallback if image is uniform (e.g. all black/background)
        return 0.0

    # Skeletonization
    # Reduces the binary fibers to 1-pixel wide lines
    # This quantifies the extent of the network independent of fiber thickness
    skeleton = skeletonize(binary_tubulin)
    
    # 4. Quantification per Cell
    
    # Get unique cell labels (excluding background 0)
    cell_ids = np.unique(cell_mask)
    cell_ids = cell_ids[cell_ids > 0]
    
    if len(cell_ids) == 0:
        return 0.0

    # Calculate skeleton length per cell
    # We sum the boolean skeleton pixels within each cell's region
    # ndimage.sum is efficient for this: sum(input, labels, index)
    # input: skeleton (converted to float), labels: cell_mask, index: cell_ids
    skeleton_pixels = ndimage.sum(skeleton.astype(np.float32), cell_mask, index=cell_ids)
    
    # Ensure we have an array (ndimage.sum can return scalar if single index)
    if np.isscalar(skeleton_pixels):
        skeleton_pixels = np.array([skeleton_pixels])
    else:
        skeleton_pixels = np.array(skeleton_pixels)
        
    # 5. Aggregate
    # Calculate mean length across all cells
    if len(skeleton_pixels) == 0:
        return 0.0
        
    result = np.mean(skeleton_pixels)
    
    return float(result)
