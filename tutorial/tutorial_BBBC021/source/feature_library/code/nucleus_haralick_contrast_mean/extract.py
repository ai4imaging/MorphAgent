def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk
    from scipy.ndimage import gaussian_filter

    # Convert to appropriate array type
    # Dataset is (512, 512, 3), uint8. Channel 2 is DAPI (Nucleus).
    img = np.asarray(img)
    
    # Handle dimensionality and extract DAPI channel
    # The dataset description specifies (Height, Width, Channels) = (512, 512, 3)
    # Channel 2 is DAPI (Blue)
    dapi = None
    if img.ndim == 3:
        if img.shape[2] == 3: # (H, W, C)
            dapi = img[:, :, 2]
        elif img.shape[0] == 3: # (C, H, W) - fallback for potential transpose
            dapi = img[2, :, :]
    elif img.ndim == 2:
        dapi = img # Fallback if single channel passed
        
    if dapi is None:
        return 0.0

    # Normalize and Quantize
    # Haralick features are sensitive to the number of gray levels.
    # Calculating GLCM on 256 levels for small nuclear regions results in sparse matrices.
    # We reduce the dynamic range to 64 levels to make the GLCM dense and robust.
    n_levels = 64
    
    # Ensure we are working with float for calculation then cast back
    dapi_float = dapi.astype(np.float32)
    
    # Binning: 0-255 -> 0-63
    # Factor = 256 / 64 = 4
    dapi_quantized = (dapi_float / 4.0).astype(np.uint8)
    dapi_quantized = np.clip(dapi_quantized, 0, n_levels - 1)

    # Handle segmentation masks
    mask = None
    if len(segmentation_masks) > 0:
        # Try to find a valid mask in the arguments
        for m in segmentation_masks:
            if m is not None and isinstance(m, np.ndarray):
                # Check if mask dimensions match image (H, W)
                if m.shape[-2:] == dapi.shape[-2:] or m.shape[:2] == dapi.shape[:2]:
                    mask = m
                    # Handle if mask has channel dim (e.g. H, W, 1)
                    if mask.ndim == 3:
                        mask = mask.squeeze()
                    break
    
    # Fallback: Generate segmentation if no mask provided
    # This is critical as the feature depends on nuclear regions
    if mask is None:
        try:
            # Preprocessing: Gaussian blur to reduce noise
            blurred = gaussian_filter(dapi_float, sigma=2.0)
            # Thresholding: Otsu's method
            thresh = threshold_otsu(blurred)
            binary = blurred > thresh
            # Morphological cleaning: Remove small noise and separate touching cells slightly
            binary = binary_opening(binary, disk(3))
            # Labeling: Create instance mask
            mask = label(binary)
        except Exception:
            return 0.0
    else:
        # Ensure mask is labeled (integers), not just binary
        # If max label is 1, it's likely a binary mask, so we label connected components
        if mask.dtype == bool or mask.max() <= 1:
            mask = label(mask)

    # Feature Extraction
    # We use regionprops to iterate over each nucleus
    props = regionprops(mask, intensity_image=dapi_quantized)
    
    contrasts = []
    
    # GLCM parameters
    # Distance 1 pixel: captures fine texture
    distances = [1]
    # 4 directions (0, 45, 90, 135 degrees) for rotational invariance
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    for prop in props:
        # Filter noise/small artifacts (e.g., < 50 pixels)
        if prop.area < 50:
            continue
            
        # Extract the bounding box of the quantized intensity image
        # Note: We use the rectangular bounding box. While this may include some background
        # pixels, it is the standard approach for GLCM in skimage to avoid complex masking
        # artifacts (like zero-padding creating high-contrast edges).
        roi = prop.image_intensity
        
        # Skip if ROI is too small (need at least 2 pixels for distance 1)
        if roi.shape[0] < 2 or roi.shape[1] < 2:
            continue

        try:
            # Compute GLCM
            # levels=64 matches our quantization
            glcm = graycomatrix(roi, distances=distances, angles=angles, 
                                levels=n_levels, symmetric=True, normed=True)
            
            # Compute Contrast
            # Returns (n_distances, n_angles) array
            contrast_mat = graycoprops(glcm, 'contrast')
            
            # Average contrast across all angles for this cell to get a rotation-invariant metric
            cell_contrast = np.mean(contrast_mat)
            contrasts.append(cell_contrast)
        except Exception:
            continue

    # Aggregation
    # Return the mean contrast across all nuclei in the image
    if len(contrasts) == 0:
        return 0.0
        
    result = np.mean(contrasts)
    return float(result)
