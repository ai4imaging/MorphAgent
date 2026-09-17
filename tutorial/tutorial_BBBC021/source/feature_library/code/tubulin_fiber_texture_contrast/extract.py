def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.filters import threshold_otsu
    from skimage.util import img_as_ubyte
    
    # 1. Input Validation and Channel Selection
    # Dataset is (512, 512, 3) uint8. Channel 1 is Tubulin (Green).
    if img is None or img.ndim < 3 or img.shape[2] < 2:
        return 0.0
        
    # Extract Tubulin channel (Channel 1)
    tubulin_channel = img[:, :, 1]
    
    # 2. Preprocessing and Quantization
    # GLCM calculation is expensive and sensitive to noise on 256 levels.
    # We quantize the image to fewer levels (e.g., 64) to make the texture features more robust
    # and computation faster.
    n_levels = 64
    
    # Normalize to 0-1 range first to handle potential dtype variations, then scale to bins
    if tubulin_channel.dtype == np.uint8:
        # Integer division is fast for uint8
        # 256 / 64 = 4. So we divide by 4.
        quantized_img = (tubulin_channel // 4).astype(np.uint8)
    else:
        # Fallback for other types
        min_val, max_val = tubulin_channel.min(), tubulin_channel.max()
        if max_val > min_val:
            norm_img = (tubulin_channel - min_val) / (max_val - min_val)
            quantized_img = (norm_img * (n_levels - 1)).astype(np.uint8)
        else:
            return 0.0

    # 3. Masking / ROI Definition
    # We need to compute texture only within the cells to avoid the large black background 
    # dominating the statistics (specifically the (0,0) entry of the GLCM).
    
    mask = None
    if len(segmentation_masks) > 0:
        # If masks are provided, combine them to get a total cellular area mask
        # Assuming masks are label matrices where 0 is background
        combined_mask = np.zeros(tubulin_channel.shape, dtype=bool)
        for m in segmentation_masks:
            if m.shape == tubulin_channel.shape:
                combined_mask = combined_mask | (m > 0)
        
        if np.any(combined_mask):
            mask = combined_mask
            
    # Fallback: If no valid masks provided, generate one using Otsu thresholding
    if mask is None:
        # Simple background segmentation
        try:
            thresh = threshold_otsu(tubulin_channel)
            mask = tubulin_channel > thresh
        except Exception:
            # If image is uniform (e.g. all black), otsu fails
            return 0.0

    # If mask is empty (no cells), return 0
    if not np.any(mask):
        return 0.0

    # 4. Apply Mask to Quantized Image
    # We set background pixels to a specific value. However, standard GLCM computes over the whole rect.
    # Strategy: 
    # 1. Set background pixels to 0.
    # 2. Shift actual data to range 1..64 (so 0 is reserved for background).
    # 3. Compute GLCM.
    # 4. Ignore the (0,0) entry of the GLCM (background-background transitions).
    
    # Shift data to 1-based indexing for valid pixels
    # Ensure we don't overflow; n_levels was 64, so max val is 63. +1 makes it 64. 
    # We need 65 levels total (0..64).
    roi_img = quantized_img.copy()
    roi_img = roi_img + 1 
    roi_img[~mask] = 0 # Set background to 0
    
    final_levels = n_levels + 1 # 0 is background, 1..64 are intensities

    # 5. Compute GLCM
    # Distances: 1 pixel (capture fine fiber texture)
    # Angles: 0, 45, 90, 135 degrees (isotropic)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    try:
        # Calculate unnormalized GLCM first
        glcm = graycomatrix(roi_img, distances=distances, angles=angles, 
                            levels=final_levels, symmetric=True, normed=False)
    except ValueError:
        return 0.0

    # 6. Background Correction
    # The entry (0,0) corresponds to background-background pairs. We remove this influence.
    # We also remove (0, x) and (x, 0) pairs if we strictly want "within-cell" texture,
    # but often boundary texture is useful. 
    # However, strictly speaking, texture *contrast* inside the cell shouldn't be affected by the 
    # cell boundary contrast against black background.
    # To measure pure fiber texture, we should ignore transitions involving the background label (0).
    
    # Slice the GLCM to exclude the 0-th row and 0-th column
    # This leaves us with the GLCM of the cellular region only (levels 1..64)
    glcm_cellular = glcm[1:, 1:, :, :]
    
    # Check if we have any valid transitions left
    if glcm_cellular.sum() == 0:
        return 0.0
        
    # Normalize the cropped GLCM so probabilities sum to 1
    glcm_norm = glcm_cellular / glcm_cellular.sum(axis=(0, 1), keepdims=True)
    
    # 7. Compute Contrast
    # Contrast = sum_{i,j} |i-j|^2 * p(i,j)
    # High contrast = large intensity differences between neighbors (e.g., bright fibers vs dark cytoplasm)
    # Low contrast = smooth meshwork
    props = graycoprops(glcm_norm, 'contrast')
    
    # Average over all angles and distances to get a single scalar
    result = np.mean(props)
    
    return float(result)
