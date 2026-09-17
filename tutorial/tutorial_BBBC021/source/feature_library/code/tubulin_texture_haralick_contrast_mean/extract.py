def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, disk

    # 1. Data Loading and Validation
    # Ensure image is numpy array
    img = np.asarray(img)
    
    # Handle dimensionality
    # Expected shape: (512, 512, 3)
    if img.ndim == 3 and img.shape[2] == 3:
        # Extract Tubulin channel (Channel 1: Green)
        tubulin_channel = img[:, :, 1]
    elif img.ndim == 2:
        # Fallback if single channel passed (unlikely based on spec, but safe)
        tubulin_channel = img
    else:
        # Unexpected format
        return 0.0

    # Ensure uint8 for GLCM
    if tubulin_channel.dtype != np.uint8:
        # Normalize to 0-255 if not already
        if tubulin_channel.max() <= 1.0:
            tubulin_channel = (tubulin_channel * 255).astype(np.uint8)
        else:
            # Clip and cast
            tubulin_channel = np.clip(tubulin_channel, 0, 255).astype(np.uint8)

    # 2. Mask Generation / Selection
    mask = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Use the first available mask (assuming it covers the cells/cytoplasm)
        # Masks are typically labeled integers (0=bg, 1..N=cells)
        input_mask = segmentation_masks[0]
        if input_mask is not None and input_mask.shape == tubulin_channel.shape:
            mask = input_mask > 0

    # Fallback: Generate mask if none provided or invalid
    if mask is None:
        # Calculate Otsu threshold on the tubulin channel itself
        # This separates signal (cells) from background
        try:
            thresh = threshold_otsu(tubulin_channel)
            mask = tubulin_channel > thresh
            # Clean up the mask slightly to fill holes
            mask = binary_closing(mask, disk(3))
        except Exception:
            # Fallback for completely empty/black images where otsu fails
            return 0.0

    # 3. Apply Mask
    # If the mask is empty (no cells), return 0.0
    if np.sum(mask) == 0:
        return 0.0

    # To compute texture ONLY within the cells, we have a challenge with GLCM on rectangular arrays.
    # Standard approach: Mask the image (set background to 0).
    # Note: This introduces artificial contrast at the cell boundaries (transition to 0).
    # However, for "mean contrast" over the whole cell population, this is a standard approximation.
    masked_tubulin = tubulin_channel.copy()
    masked_tubulin[~mask] = 0

    # 4. Compute Haralick Contrast
    # Parameters:
    # - distances=[1]: Pixel adjacency (captures fine texture like microtubules)
    # - angles=[0, 45, 90, 135]: Rotation invariance
    # - levels=256: Standard for uint8
    
    # Optimization: If the image is very sparse, we could crop to the bounding box of the mask,
    # but 512x512 is fast enough to process fully.
    
    try:
        # Compute GLCM
        # We use the masked image. The background (0) is treated as a gray level.
        # This means 0-0 transitions (background) and X-0 transitions (edges) are included.
        glcm = graycomatrix(masked_tubulin, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                            levels=256, symmetric=True, normed=True)
        
        # Compute Contrast
        # Contrast measures the intensity contrast between a pixel and its neighbor over the whole image.
        contrast_matrix = graycoprops(glcm, 'contrast')
        
        # 5. Refinement (Optional but recommended for masked images)
        # The standard GLCM includes the massive amount of background (0,0) co-occurrences.
        # While 'contrast' weights (i-j)^2, so (0-0)^2 is 0 and doesn't add to the sum,
        # the normalization of the GLCM (dividing by total pairs) IS affected by the background size.
        # However, 'graycoprops' returns the weighted sum.
        # If the feature is strictly "Haralick Contrast", we return the mean of the computed properties.
        
        # Average over the 4 angles to get a rotation-invariant feature
        mean_contrast = np.mean(contrast_matrix)
        
        return float(mean_contrast)

    except Exception:
        return 0.0
