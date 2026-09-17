def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.filters import threshold_otsu
    
    # 1. Input Validation and Channel Extraction
    # Ensure input is a numpy array
    arr = np.asarray(img)
    
    # Check dimensions. Expected: (H, W, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not 3 channels, we cannot reliably identify Tubulin (Channel 1).
        # Return 0.0 as fallback.
        return 0.0

    # Extract Tubulin Channel (Channel 1 - Green)
    # The dataset description specifies: Channel 1 = Green = Tubulin
    tubulin_channel = arr[:, :, 1]

    # 2. Preprocessing and Type Conversion
    # GLCM requires integer types. The dataset is uint8, which is perfect.
    # If for some reason it's float, we need to scale and cast.
    if np.issubdtype(tubulin_channel.dtype, np.floating):
        # Normalize to 0-255 and cast to uint8
        min_val = np.min(tubulin_channel)
        max_val = np.max(tubulin_channel)
        if max_val > min_val:
            tubulin_channel = ((tubulin_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            tubulin_channel = np.zeros_like(tubulin_channel, dtype=np.uint8)
    elif tubulin_channel.dtype != np.uint8:
        # If it's some other int type, clip and cast
        tubulin_channel = np.clip(tubulin_channel, 0, 255).astype(np.uint8)

    # 3. Region of Interest (ROI) Masking
    # We want to calculate texture primarily on the cells, not the background.
    # Background pixels (0) can skew texture statistics if they dominate.
    
    mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Combine all provided masks
        combined_mask = np.zeros(tubulin_channel.shape, dtype=bool)
        for m in segmentation_masks:
            if m is not None:
                # Ensure mask matches image shape (handle potential 2D vs 3D mismatch if any)
                if m.shape == tubulin_channel.shape:
                    combined_mask = np.logical_or(combined_mask, m > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # Fallback: If no valid mask provided, generate one using Otsu thresholding on the Tubulin channel
    if mask is None:
        try:
            # Check if image has content
            if np.max(tubulin_channel) > np.min(tubulin_channel):
                thresh = threshold_otsu(tubulin_channel)
                mask = tubulin_channel > thresh
            else:
                # Flat image, return 0 contrast
                return 0.0
        except Exception:
            # Fallback for extremely sparse images where Otsu might fail
            return 0.0

    # Apply mask: Set background to 0. 
    # Note: In GLCM, 0-0 transitions (background) contribute 0 to contrast (i-j=0),
    # but they affect the normalization. 
    # However, standard practice for simple patch-based GLCM often just masks the array.
    # A more rigorous approach computes GLCM only on mask pixels, but skimage graycomatrix
    # operates on the full rectangular array.
    # By zeroing the background, we ensure background noise doesn't create high contrast artifacts,
    # though the boundary between cell and background will create a texture edge.
    masked_img = tubulin_channel.copy()
    masked_img[~mask] = 0

    # 4. GLCM Computation
    # Parameters:
    # - distances: [1] (immediate neighbors)
    # - angles: 0, 45, 90, 135 degrees (for rotational invariance)
    # - levels: 256 (for uint8)
    try:
        glcm = graycomatrix(
            masked_img, 
            distances=[1], 
            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
            levels=256, 
            symmetric=True, 
            normed=True
        )
    except ValueError:
        # Can happen if image is empty or invalid
        return 0.0

    # 5. Feature Extraction: Contrast
    # Contrast: sum(|i-j|^2 * p(i,j))
    # Measures local intensity variation.
    # High contrast = sharp edges, fibrous structures (microtubules).
    # Low contrast = smooth, diffuse staining.
    contrast_values = graycoprops(glcm, 'contrast')

    # 6. Aggregation
    # Average the contrast across all 4 directions to get a rotation-invariant feature
    avg_contrast = np.mean(contrast_values)

    return float(avg_contrast)
