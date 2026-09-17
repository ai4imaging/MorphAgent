def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.filters import threshold_otsu

    # 1. Input Validation and Channel Selection
    # Ensure input is a numpy array
    arr = np.asarray(img)
    
    # Check dimensionality and extract Tubulin channel (Channel 1)
    # Dataset is (512, 512, 3) RGB. Channel 1 is Green/Tubulin.
    if arr.ndim == 3 and arr.shape[2] >= 2:
        tubulin_channel = arr[:, :, 1]
    elif arr.ndim == 2:
        # Fallback if only 2D image provided (unlikely given description, but safe)
        tubulin_channel = arr
    else:
        return 0.0

    # 2. Data Type Handling for GLCM
    # GLCM requires integer types. The dataset is uint8 (0-255).
    # If float, scale to 0-255. If integer but not uint8, cast.
    if np.issubdtype(tubulin_channel.dtype, np.floating):
        # Normalize to 0-255 range if float
        min_val = np.min(tubulin_channel)
        max_val = np.max(tubulin_channel)
        if max_val > min_val:
            tubulin_channel = ((tubulin_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            tubulin_channel = np.zeros_like(tubulin_channel, dtype=np.uint8)
    else:
        # Ensure it is uint8 for graycomatrix (levels=256)
        tubulin_channel = tubulin_channel.astype(np.uint8)

    # 3. ROI / Mask Generation
    # We want to compute texture only within the cells to avoid analyzing the black background.
    mask = None
    
    if len(segmentation_masks) > 0:
        # Combine all provided masks into a single binary foreground mask
        combined_mask = np.zeros(tubulin_channel.shape, dtype=bool)
        for m in segmentation_masks:
            # Handle potential shape mismatches (e.g. if mask is 3D or different size)
            if m.shape == tubulin_channel.shape:
                combined_mask = np.logical_or(combined_mask, m > 0)
            elif m.ndim == 3 and m.shape[:2] == tubulin_channel.shape:
                 # If mask is 3D (e.g. labeled volume), project max
                 combined_mask = np.logical_or(combined_mask, np.max(m, axis=2) > 0)
        
        if np.any(combined_mask):
            mask = combined_mask
    
    # Fallback: If no masks provided or mask is empty, generate one using Otsu thresholding
    if mask is None:
        # Simple background separation
        try:
            thresh = threshold_otsu(tubulin_channel)
            mask = tubulin_channel > thresh
        except Exception:
            # If image is uniform (e.g. all black), otsu fails
            return 0.0

    # 4. Apply Mask
    # Set background pixels to 0. Note: This creates a strong edge at the boundary,
    # but standard rectangular GLCM implementations usually accept this trade-off.
    # We ensure the background is 0, which is a valid gray level.
    masked_img = tubulin_channel.copy()
    masked_img[~mask] = 0

    # If the image is empty after masking, return 0
    if np.sum(masked_img) == 0:
        return 0.0

    # 5. Compute GLCM
    # Parameters:
    # - distances=[1]: Pixel adjacency for fine texture
    # - angles=[0, 45, 90, 135]: Rotation invariance (average over directions)
    # - levels=256: Standard for uint8
    try:
        glcm = graycomatrix(masked_img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                            levels=256, symmetric=True, normed=True)
    except ValueError:
        # Can happen if image contains values outside [0, levels-1]
        return 0.0

    # 6. Compute Haralick Contrast
    # Contrast measures the local variations in the gray-level co-occurrence matrix.
    # High contrast = sharp transitions (Microtubule bundles).
    # Low contrast = smooth transitions (Diffuse staining).
    contrast_matrix = graycoprops(glcm, 'contrast')
    
    # Average across all angles to get a single rotation-invariant scalar
    feature_value = np.mean(contrast_matrix)

    return float(feature_value)
