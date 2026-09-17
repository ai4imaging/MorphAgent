def extract(img, *segmentation_masks):
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import disk, opening
    from skimage.filters import threshold_otsu

    # 1. Input Validation and Preprocessing
    # Convert to float32 for precision
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality and extract DAPI channel (Channel 2 / Blue)
    # Expected shape: (H, W, 3) or (H, W)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        # Channel 2 is DAPI (Blue)
        dapi_img = arr[..., 2]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on spec, but safe)
        dapi_img = arr
    else:
        return 0.0

    # Normalize intensity to [0, 1]
    # Using robust max to handle outliers, but ensuring valid range
    vmax = np.percentile(dapi_img, 99.5) if dapi_img.size > 0 else 1.0
    if vmax <= 0:
        vmax = 1.0
    dapi_img = dapi_img / vmax
    dapi_img = np.clip(dapi_img, 0.0, 1.0)

    # 2. Mask Handling
    # We need a mask to calculate granularity only within the nuclei.
    # Calculating texture on the black background dilutes the signal.
    mask = None
    
    # Try to use provided segmentation masks
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Assume the first mask is the primary object mask (nuclei or cells)
        # Ensure mask matches image spatial dimensions
        input_mask = segmentation_masks[0]
        if input_mask.shape[:2] == dapi_img.shape[:2]:
            mask = input_mask > 0

    # Fallback: Generate mask using Otsu thresholding if no valid mask provided
    if mask is None:
        try:
            thresh = threshold_otsu(dapi_img)
            mask = dapi_img > thresh
        except Exception:
            # If thresholding fails (e.g., constant image), use all pixels
            mask = np.ones_like(dapi_img, dtype=bool)

    # If mask is empty (no objects), return 0.0
    if np.sum(mask) == 0:
        return 0.0

    # 3. Granulometry Calculation
    # Granulometry measures the size distribution of bright features (granules/foci).
    # We perform a series of morphological openings with increasing structural element sizes.
    # The "granularity" at a specific scale is the loss of intensity when opening with that size.
    
    # Define scales (radii of disk structuring elements)
    # Scales 1-5 cover fine texture (chromatin) to medium blobs (foci)
    scales = [1, 2, 3, 4, 5]
    
    spectrum_means = []
    prev_opened = dapi_img.copy()

    for r in scales:
        # Create structuring element
        selem = disk(r)
        
        # Perform morphological opening
        # Opening removes bright features smaller than the structuring element
        curr_opened = opening(dapi_img, selem)
        
        # Calculate the difference (features removed at this scale or smaller)
        # Standard granulometry spectrum is often defined as: Sum(Open_r-1) - Sum(Open_r)
        # Here we calculate the pixel-wise difference between the previous step and current step
        # to isolate features of specific size range r.
        diff_img = prev_opened - curr_opened
        
        # Calculate mean intensity of these specific features *within the mask*
        # This gives the "strength" of texture at this scale
        mean_intensity = np.mean(diff_img[mask])
        spectrum_means.append(mean_intensity)
        
        # Update previous image for next iteration
        prev_opened = curr_opened

    # 4. Compute Final Feature
    # The result is the average of the granularity scores across the measured scales.
    # This represents the "mean granularity" of the nuclei.
    if len(spectrum_means) == 0:
        return 0.0
        
    result = np.mean(spectrum_means)

    return float(result)
