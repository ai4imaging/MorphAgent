def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.filters import threshold_otsu

    # 1. Input Validation and Channel Extraction
    # Ensure input is a numpy array
    img = np.asarray(img)
    
    # Check for valid dimensions (512, 512, 3)
    if img.ndim != 3 or img.shape[2] < 2:
        return 0.0
        
    # Extract the Tubulin channel (Channel 1 - Green)
    # The dataset description specifies: Channel 0=Actin, Channel 1=Tubulin, Channel 2=DAPI
    tubulin_channel = img[..., 1]

    # 2. Data Type Handling
    # GLCM calculation requires integer types. The dataset is uint8, which is ideal.
    # If it were float, we would need to quantize it.
    if tubulin_channel.dtype != np.uint8:
        # If not uint8, normalize to 0-255 and cast
        if np.issubdtype(tubulin_channel.dtype, np.floating):
            min_val, max_val = tubulin_channel.min(), tubulin_channel.max()
            if max_val > min_val:
                tubulin_channel = ((tubulin_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                tubulin_channel = np.zeros_like(tubulin_channel, dtype=np.uint8)
        else:
            # If other integer type, clip and cast
            tubulin_channel = np.clip(tubulin_channel, 0, 255).astype(np.uint8)

    # 3. ROI Definition (Masking)
    # We want to compute texture primarily on the cellular regions, not the background noise.
    mask = None
    
    if len(segmentation_masks) > 0:
        # If segmentation masks are provided, combine them to create a foreground mask
        # Masks are typically labeled integers (0=bg, 1..N=cells)
        combined_mask = np.zeros(tubulin_channel.shape, dtype=bool)
        for m in segmentation_masks:
            if m is not None and m.shape == tubulin_channel.shape:
                combined_mask = np.logical_or(combined_mask, m > 0)
        
        if np.any(combined_mask):
            mask = combined_mask
    
    # Fallback: If no masks provided or mask is empty, use Otsu thresholding on the tubulin channel itself
    if mask is None:
        # Check if image is not empty/flat
        if np.max(tubulin_channel) > np.min(tubulin_channel):
            try:
                thresh = threshold_otsu(tubulin_channel)
                mask = tubulin_channel > thresh
            except Exception:
                # Fallback for extremely low contrast images where otsu might fail
                mask = np.ones_like(tubulin_channel, dtype=bool)
        else:
            # Flat image, no texture
            return 0.0

    # 4. Apply Mask
    # For GLCM, we typically want to ignore the background. However, standard implementations
    # compute GLCM on the rectangular array. A common approach is to set background to 0.
    # Note: This creates a strong edge at the boundary of the mask, which affects texture.
    # However, calculating GLCM only on pixels within an irregular ROI is complex.
    # A robust approximation is to zero out the background.
    masked_tubulin = tubulin_channel.copy()
    masked_tubulin[~mask] = 0

    # 5. Compute GLCM
    # Parameters:
    # - distances: [1, 3] to capture fine and slightly coarser texture
    # - angles: 0, 45, 90, 135 degrees for rotational invariance
    # - levels: 256 (for uint8)
    distances = [1, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    try:
        glcm = graycomatrix(masked_tubulin, distances=distances, angles=angles, 
                            levels=256, symmetric=True, normed=True)
    except ValueError:
        return 0.0

    # 6. Compute Haralick Contrast
    # Contrast measures the local intensity variation.
    # High contrast = sharp edges/transitions (e.g., distinct microtubules).
    # Low contrast = smooth/blurry regions.
    contrast_matrix = graycoprops(glcm, 'contrast')
    
    # 7. Aggregation
    # Average the contrast across all specified distances and angles to get a single scalar
    # representing the overall texture "roughness" of the tubulin network.
    feature_value = np.mean(contrast_matrix)

    return float(feature_value)
