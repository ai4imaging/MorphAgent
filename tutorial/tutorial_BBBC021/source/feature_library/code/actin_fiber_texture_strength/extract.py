def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type
    # Image is (512, 512, 3), uint8. Channel 0 is Actin (Red).
    # We need integer types for GLCM, so keep as uint8 or convert to int
    if img.ndim != 3 or img.shape[2] < 1:
        return 0.0
        
    # 1. Select the Actin Channel (Channel 0)
    actin_channel = img[:, :, 0]
    
    # 2. Handle Segmentation / Masking
    # We need to define the region of interest (ROI) to avoid calculating texture on the black background.
    # If masks are provided, combine them. If not, generate a foreground mask.
    mask = None
    if len(segmentation_masks) > 0:
        # Combine all masks (assuming labeled masks where >0 is cell)
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        for m in segmentation_masks:
            if m is not None and m.shape == actin_channel.shape:
                combined_mask = combined_mask | (m > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # Fallback if no valid masks provided or mask is empty
    if mask is None:
        # Create a simple foreground mask using Otsu thresholding
        # Check if image is not completely flat
        if np.min(actin_channel) == np.max(actin_channel):
            return 0.0
        thresh = threshold_otsu(actin_channel)
        mask = actin_channel > thresh

    # If mask is still empty (e.g. extremely dark image), return 0
    if not np.any(mask):
        return 0.0

    # 3. Preprocessing for GLCM
    # GLCM is computationally expensive on 256 levels. Binning to 64 levels is standard practice.
    # We mask the image by setting background to -1 (or a distinct value) to handle it carefully,
    # but skimage graycomatrix doesn't support masked arrays or negative values easily.
    # Strategy:
    # 1. Quantize the ROI pixels to 0-63.
    # 2. Set background pixels to a distinct value (e.g., 64) that is outside the ROI range.
    # 3. Compute GLCM for levels=65.
    # 4. Slice the GLCM to only include 0-63 interactions (ignore interactions with background label 64).
    
    n_bins = 64
    # Normalize ROI to 0-1 range first
    roi_pixels = actin_channel[mask]
    if roi_pixels.size == 0:
        return 0.0
        
    p_min, p_max = roi_pixels.min(), roi_pixels.max()
    if p_max - p_min == 0:
        return 0.0
        
    # Binning logic: map [min, max] to [0, n_bins-1]
    # We apply this transformation to the whole image array
    # Use float for calculation then cast to int
    binned_img = np.floor((actin_channel.astype(np.float32) - p_min) / (p_max - p_min) * (n_bins - 1e-5)).astype(np.uint8)
    
    # Set background to n_bins (64)
    # This ensures background pixels don't get mixed into the 0-63 texture bins
    binned_img[~mask] = n_bins
    
    # 4. Compute GLCM
    # We use distances [1, 3] to capture fine fibers and slightly coarser bundles
    # We average over 4 angles for rotational invariance
    distances = [1, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    levels = n_bins + 1 # 0 to 64
    
    glcm = graycomatrix(binned_img, distances=distances, angles=angles, levels=levels, symmetric=True, normed=False)
    
    # 5. Filter GLCM to exclude background interactions
    # The GLCM is shape (levels, levels, num_distances, num_angles)
    # We only care about the sub-matrix [0:64, 0:64, :, :]
    # This effectively ignores any pixel pair where one or both pixels were background (label 64)
    glcm_roi = glcm[0:n_bins, 0:n_bins, :, :]
    
    # Normalize the valid sub-GLCMs so they sum to 1
    # Sum over the first two dimensions (i, j) for each (distance, angle) pair
    glcm_sums = glcm_roi.sum(axis=(0, 1), keepdims=True)
    
    # Avoid division by zero if a specific direction has no valid pairs
    glcm_sums[glcm_sums == 0] = 1.0
    glcm_normed = glcm_roi / glcm_sums

    # 6. Calculate Contrast
    # Contrast = sum_{i,j} P[i,j] * (i-j)^2
    # We can use skimage's graycoprops, but we need to pass the manually normalized/sliced GLCM
    # graycoprops expects the full GLCM structure, but since we sliced it, we can just pass the slice
    # provided we interpret the output correctly.
    # However, graycoprops might re-normalize or expect specific shapes. 
    # It's safer to implement the simple contrast formula manually on the numpy array.
    
    # Create (i-j)^2 matrix
    # shape (n_bins, n_bins)
    rows, cols = np.indices((n_bins, n_bins))
    weights = (rows - cols) ** 2
    weights = weights[:, :, np.newaxis, np.newaxis] # broadcast to (n_bins, n_bins, n_dists, n_angles)
    
    contrast_matrix = np.sum(glcm_normed * weights, axis=(0, 1))
    
    # 7. Aggregate
    # Average over all distances and angles to get a single scalar "strength"
    result = np.mean(contrast_matrix)

    return float(result)
