def extract(img, *segmentation_masks):
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu

    # 1. Extract the DAPI channel (Channel 2 based on dataset description)
    # Input is (512, 512, 3), DAPI is index 2 (Blue)
    if img.ndim == 3 and img.shape[2] >= 3:
        dapi_channel = img[:, :, 2]
    elif img.ndim == 2:
        # Fallback if single channel passed, though unlikely given description
        dapi_channel = img
    else:
        return 0.0

    # Ensure uint8 for GLCM calculation logic
    if dapi_channel.dtype != np.uint8:
        # Normalize to 0-255 if not already
        dapi_min, dapi_max = dapi_channel.min(), dapi_channel.max()
        if dapi_max > dapi_min:
            dapi_channel = ((dapi_channel - dapi_min) / (dapi_max - dapi_min) * 255).astype(np.uint8)
        else:
            dapi_channel = dapi_channel.astype(np.uint8)

    # 2. Handle Segmentation
    # If masks are provided, use the first one (assuming it's the nuclei/cell mask).
    # If not, generate a mask using Otsu thresholding on the DAPI channel.
    labeled_mask = None
    
    if segmentation_masks and len(segmentation_masks) > 0:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == dapi_channel.shape[:2]:
            # If mask is already labeled (int > 1), use it. If binary, label it.
            if np.issubdtype(mask_input.dtype, np.integer) and mask_input.max() > 1:
                labeled_mask = mask_input
            else:
                labeled_mask = label(mask_input > 0)
    
    # Fallback segmentation if no valid mask provided
    if labeled_mask is None:
        try:
            thresh = threshold_otsu(dapi_channel)
            binary_mask = dapi_channel > thresh
            labeled_mask = label(binary_mask)
        except Exception:
            # If thresholding fails (e.g. constant image), return 0
            return 0.0

    # 3. Compute Haralick Contrast per Nucleus
    regions = regionprops(labeled_mask, intensity_image=dapi_channel)
    contrasts = []

    # GLCM parameters
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    for region in regions:
        # Skip very small regions that might be noise
        if region.area < 10:
            continue

        # Extract the bounding box intensity image
        # region.image is the binary mask of the object in the bounding box
        # region.intensity_image is the intensity values in the bounding box
        roi_intensity = region.intensity_image
        roi_mask = region.image

        # Prepare image for GLCM:
        # We want to ignore the background (0) in the bounding box so it doesn't affect texture.
        # Strategy: Shift valid pixels by +1 (range 1-256), keep background as 0.
        # This requires a larger type than uint8 to hold 256 safely if max was 255.
        # However, graycomatrix expects input values to be indices.
        # We will use a max level of 256 (indices 0-256).
        
        # Create a working array
        work_arr = np.zeros(roi_intensity.shape, dtype=np.uint16)
        
        # Copy intensities where mask is True, adding 1 to shift away from 0
        work_arr[roi_mask] = roi_intensity[roi_mask].astype(np.uint16) + 1
        
        # Compute GLCM
        # levels=257 because values are 0..256
        try:
            glcm = graycomatrix(work_arr, distances=distances, angles=angles, 
                                levels=257, symmetric=True, normed=False)
        except ValueError:
            continue

        # 4. Clean GLCM (Remove Background)
        # The row/col at index 0 corresponds to the background.
        # We remove it to calculate texture ONLY within the nucleus.
        glcm_cut = glcm[1:, 1:, :, :] # Shape becomes (256, 256, n_dist, n_angles)
        
        # Re-normalize
        glcm_sum = glcm_cut.sum(axis=(0, 1), keepdims=True)
        # Avoid division by zero if region was empty or uniform
        glcm_sum[glcm_sum == 0] = 1
        glcm_norm = glcm_cut / glcm_sum

        # 5. Calculate Contrast
        # graycoprops expects the full GLCM structure, but we sliced it.
        # However, graycoprops logic is simple: sum(P[i,j] * (i-j)^2).
        # We can implement it manually for the sliced GLCM to be safe and accurate.
        
        # Create weight matrix (i-j)^2
        n_levels = glcm_norm.shape[0] # 256
        rows, cols = np.ogrid[:n_levels, :n_levels]
        weights = (rows - cols) ** 2
        weights = weights[:, :, np.newaxis, np.newaxis] # broadcast to (256, 256, 1, 1)
        
        # Calculate contrast: sum(weights * probabilities)
        contrast_val = np.sum(glcm_norm * weights)
        
        # Average over angles and distances (though we sum over them implicitly if we don't keep dims, 
        # but here we want the mean scalar for this cell)
        # The sum above summed over i,j. We now have shape (1, 1, n_dist, n_angles) effectively if we didn't sum axes.
        # Actually np.sum(glcm_norm * weights) returns a single scalar sum because weights broadcasts.
        # But wait, we need to average over the 4 angles.
        # Let's do it explicitly:
        
        layer_contrasts = []
        for d_idx in range(len(distances)):
            for a_idx in range(len(angles)):
                p = glcm_norm[:, :, d_idx, a_idx]
                # weights is (256, 256, 1, 1), slice it to (256, 256)
                w = (rows - cols) ** 2
                val = np.sum(p * w)
                layer_contrasts.append(val)
        
        mean_contrast_cell = np.mean(layer_contrasts)
        contrasts.append(mean_contrast_cell)

    # 6. Aggregate
    if not contrasts:
        return 0.0
        
    return float(np.mean(contrasts))
