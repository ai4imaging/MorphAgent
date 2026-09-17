def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_erosion, disk

    # 1. Data Preparation
    # Convert input to numpy array if not already
    arr = np.asarray(img)
    
    # Check dimensionality and extract Tubulin channel (Channel 1 - Green)
    # Expected shape is (H, W, C) = (512, 512, 3)
    if arr.ndim == 3 and arr.shape[2] == 3:
        tubulin_channel = arr[:, :, 1]  # Extract Green channel
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on spec, but safe)
        tubulin_channel = arr
    else:
        return 0.0

    # Ensure uint8 for GLCM
    if tubulin_channel.dtype != np.uint8:
        # Normalize to 0-255 if not uint8
        min_val, max_val = tubulin_channel.min(), tubulin_channel.max()
        if max_val > min_val:
            tubulin_channel = ((tubulin_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            tubulin_channel = tubulin_channel.astype(np.uint8)

    # 2. Define Region of Interest (Cytoplasm)
    # We want to measure texture in the cytoplasm, excluding background and ideally nuclei
    
    cytoplasm_mask = None
    
    # Check if segmentation masks are provided
    # Standard expectation: mask 0 = nuclei, mask 1 = cells (if available)
    # Or just mask 0 = cells. We need to be adaptive.
    
    has_masks = len(segmentation_masks) > 0
    
    if has_masks:
        try:
            # Try to identify cell and nuclei masks
            # If multiple masks, assume the larger total area is cell, smaller is nucleus
            masks = [np.asarray(m) for m in segmentation_masks]
            
            if len(masks) >= 2:
                # Heuristic: Nuclei are usually smaller and contained within cells
                areas = [np.sum(m > 0) for m in masks]
                sorted_indices = np.argsort(areas)
                
                nuclei_mask = masks[sorted_indices[0]] > 0
                cell_mask = masks[sorted_indices[-1]] > 0
                
                # Cytoplasm = Cell - Nuclei
                cytoplasm_mask = np.logical_and(cell_mask, ~nuclei_mask)
            elif len(masks) == 1:
                # Only one mask, assume it's the cell/foreground
                cytoplasm_mask = masks[0] > 0
        except Exception:
            # Fallback to image-based segmentation if mask processing fails
            has_masks = False

    if not has_masks or cytoplasm_mask is None or np.sum(cytoplasm_mask) == 0:
        # Fallback: Create mask using Otsu thresholding on the tubulin channel
        # This separates signal (cells) from background
        try:
            thresh = threshold_otsu(tubulin_channel)
            cytoplasm_mask = tubulin_channel > thresh
            # Erode slightly to remove noisy edges
            cytoplasm_mask = binary_erosion(cytoplasm_mask, disk(1))
        except Exception:
            # If image is uniform (e.g., all black), otsu fails
            return 0.0

    # If mask is still empty after fallback, return 0
    if np.sum(cytoplasm_mask) == 0:
        return 0.0

    # 3. Intensity Quantization (Binning)
    # GLCM on 256 levels is slow and sparse. Binning to 32 levels is standard for texture.
    # We map background to 0, and ROI pixels to 1-32.
    
    n_bins = 32
    # Get pixels inside the mask
    roi_pixels = tubulin_channel[cytoplasm_mask]
    
    if roi_pixels.size == 0:
        return 0.0
        
    # Quantize ROI pixels
    # Use percentile to be robust against outliers
    p_min, p_max = np.percentile(roi_pixels, 1), np.percentile(roi_pixels, 99)
    if p_max <= p_min:
        return 0.0 # No contrast possible
        
    # Clip and scale to 0-(n_bins-1)
    roi_quantized = np.clip((roi_pixels - p_min) / (p_max - p_min), 0, 1)
    roi_quantized = (roi_quantized * (n_bins - 1)).astype(np.uint8) + 1 # Shift to 1-32
    
    # Create a full image for GLCM calculation
    # Background is 0, ROI is 1-32
    quantized_img = np.zeros_like(tubulin_channel, dtype=np.uint8)
    quantized_img[cytoplasm_mask] = roi_quantized

    # 4. GLCM Computation
    # Compute GLCM
    # distances=[1] for immediate neighbors (fine texture)
    # angles=[0, 45, 90, 135] for rotational invariance
    # levels=n_bins + 1 (0 for background + 32 bins)
    try:
        glcm = graycomatrix(quantized_img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                            levels=n_bins + 1, symmetric=True, normed=False)
    except ValueError:
        return 0.0

    # 5. Mask Handling in GLCM
    # The GLCM currently includes transitions involving background (0).
    # We must remove row 0 and col 0 to analyze only texture *within* the cytoplasm.
    # Slicing: [1:, 1:, :, :]
    roi_glcm = glcm[1:, 1:, :, :]
    
    # Check if we have any valid transitions
    if np.sum(roi_glcm) == 0:
        return 0.0
        
    # Normalize the sliced GLCM so probabilities sum to 1
    # graycoprops expects a normalized GLCM if we want standard interpretation, 
    # but skimage's graycoprops handles unnormalized counts for 'contrast' correctly 
    # (it sums P[i,j] * (i-j)^2). However, to be scale-invariant regarding ROI size, 
    # we must normalize manually here because we sliced the matrix.
    
    roi_glcm_sum = np.sum(roi_glcm, axis=(0, 1), keepdims=True)
    # Avoid division by zero
    roi_glcm_norm = np.divide(roi_glcm, roi_glcm_sum, where=roi_glcm_sum!=0)

    # 6. Feature Calculation: Contrast
    # Contrast measures the local intensity variation.
    # High contrast = distinct fibers/bundles. Low contrast = diffuse/smooth.
    contrast_props = graycoprops(roi_glcm_norm, 'contrast')
    
    # Average across all 4 angles to get an isotropic measure
    texture_strength = np.mean(contrast_props)

    return float(texture_strength)
