def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import skeletonize
    from skimage.filters import threshold_otsu, gaussian
    
    # 1. Input Validation and Setup
    # Ensure image is float32 for processing
    img_arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensionality: Expecting (H, W, C) or (H, W)
    if img_arr.ndim == 3:
        # Dataset description says Channel 1 is Tubulin (Green)
        # Shape is (512, 512, 3)
        if img_arr.shape[2] >= 2:
            tubulin = img_arr[:, :, 1]
        else:
            # Fallback if channels are missing, use the last available or mean
            tubulin = np.mean(img_arr, axis=2)
    elif img_arr.ndim == 2:
        tubulin = img_arr
    else:
        return 0.0

    # Normalize intensity to [0, 1]
    # Robust normalization using percentiles to handle outliers
    p_min, p_max = np.percentile(tubulin, (1, 99))
    if p_max > p_min:
        tubulin = (tubulin - p_min) / (p_max - p_min)
    else:
        tubulin = tubulin - p_min # Should be 0 array
    tubulin = np.clip(tubulin, 0.0, 1.0)

    # 2. Handle Segmentation Masks
    # We need a mask to define "within each cell".
    # If multiple masks are provided, we prioritize the first one (often whole-cell or primary segmentation).
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        cell_mask = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if cell_mask.shape != tubulin.shape:
            # Simple resize or crop logic is risky without cv2.resize, 
            # but usually masks match. If not, we can't proceed reliably per-cell.
            # Fallback: Treat whole image as one region if shapes mismatch drastically
            if cell_mask.ndim == 2:
                 # Try to use it if it's just a channel mismatch issue
                 pass
            else:
                return 0.0
    else:
        # If no segmentation is provided, we cannot compute "mean per cell".
        # However, to return a valid float, we could treat the whole image as one "cell"
        # or try to generate a crude mask. Given the prompt asks for "within each cell",
        # returning 0.0 or NaN is appropriate if no cells are defined.
        # Let's try to generate a simple foreground mask based on intensity to be robust.
        try:
            thresh = threshold_otsu(tubulin)
            cell_mask = (tubulin > thresh).astype(np.int32)
            cell_mask = ndimage.label(cell_mask)[0]
        except Exception:
            return 0.0

    # Get unique cell labels (excluding background 0)
    unique_labels = np.unique(cell_mask)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) == 0:
        return 0.0

    # 3. Preprocessing for Skeletonization
    # Smooth the image to reduce noise-induced branches in the skeleton
    # Sigma=1.0 is a reasonable default for 512x512 microscopy images
    tubulin_smooth = gaussian(tubulin, sigma=1.0)

    # Determine threshold for tubulin structure
    # We calculate Otsu threshold only on the pixels belonging to cells (foreground)
    # to avoid the large black background skewing the threshold.
    mask_bool = cell_mask > 0
    if not np.any(mask_bool):
        return 0.0
        
    foreground_pixels = tubulin_smooth[mask_bool]
    
    try:
        thresh_val = threshold_otsu(foreground_pixels)
    except Exception:
        # Fallback if image is uniform
        thresh_val = 0.5

    # Create binary tubulin structure map
    # We require the pixel to be both above threshold AND inside a cell mask
    binary_tubulin = (tubulin_smooth > thresh_val) & mask_bool

    # 4. Skeletonization
    # Skeletonize the global binary image. This is more efficient than looping per cell crop.
    # skeletonize returns a boolean array
    skeleton = skeletonize(binary_tubulin)

    # 5. Quantification per Cell
    # We want the sum of skeleton pixels for each label.
    # ndimage.sum allows us to sum the values of 'skeleton' (0 or 1) over the regions defined by 'cell_mask'.
    # The index argument specifies which labels to compute the sum for.
    
    # Convert skeleton to float or int for summation
    skeleton_int = skeleton.astype(np.float32)
    
    # Sum skeleton pixels per cell
    skeleton_lengths = ndimage.sum(skeleton_int, labels=cell_mask, index=unique_labels)

    # 6. Compute Mean
    if len(skeleton_lengths) == 0:
        return 0.0
        
    mean_length = np.mean(skeleton_lengths)

    return float(mean_length)
