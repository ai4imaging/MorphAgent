def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops

    # --- 1. Data Loading and Preprocessing ---
    # Convert to float32 for processing
    img = np.asarray(img, dtype=np.float32)

    # Handle dimensions: Ensure (H, W, C)
    # Dataset spec: (512, 512, 3). 
    # Check for Channels-First (C, H, W) just in case and transpose if necessary
    if img.ndim == 3:
        if img.shape[0] < 10 and img.shape[2] > 10:
            img = np.transpose(img, (1, 2, 0))
    
    # Extract Tubulin Channel (Channel 1 - Green)
    # BBBC021: Ch0=Actin, Ch1=Tubulin, Ch2=DAPI
    if img.ndim == 3 and img.shape[2] >= 2:
        tubulin = img[:, :, 1]
    elif img.ndim == 3:
        # Fallback if only 1 channel exists but has 3 dims
        tubulin = img[:, :, 0]
    else:
        # 2D case
        tubulin = img

    # Normalize Tubulin Channel
    # Check for zero variance to avoid NaNs
    if np.max(tubulin) == np.min(tubulin):
        return 0.0
        
    # Robust percentile normalization
    p1, p99 = np.percentile(tubulin, (1, 99))
    if p99 > p1:
        tubulin = (tubulin - p1) / (p99 - p1)
    else:
        tubulin = (tubulin - p1) 
    tubulin = np.clip(tubulin, 0, 1)

    # --- 2. Structure Tensor Calculation ---
    # The alignment index is derived from the coherence of the structure tensor.
    # High coherence = aligned fibers (anisotropic). Low coherence = meshwork (isotropic).
    
    # Compute gradients
    # np.gradient returns [gradient_y, gradient_x] for 2D array
    gy, gx = np.gradient(tubulin)

    # Compute tensor components
    gxx = gx * gx
    gyy = gy * gy
    gxy = gx * gy

    # Smooth tensor components to integrate local neighborhood
    # Sigma determines the scale of the texture analysis (fiber scale)
    sigma = 2.0
    gxx_smooth = ndimage.gaussian_filter(gxx, sigma)
    gyy_smooth = ndimage.gaussian_filter(gyy, sigma)
    gxy_smooth = ndimage.gaussian_filter(gxy, sigma)

    # Compute Coherence (Anisotropy)
    # Formula: Coherence = sqrt((Jxx - Jyy)^2 + 4*Jxy^2) / (Jxx + Jyy)
    # This ranges from 0 (isotropic) to 1 (perfectly oriented)
    
    trace = gxx_smooth + gyy_smooth
    diff = gxx_smooth - gyy_smooth
    
    # Avoid division by zero
    # We only compute coherence where there is sufficient signal (trace > epsilon)
    epsilon = 1e-7
    mask_valid = trace > epsilon
    
    coherence_map = np.zeros_like(tubulin)
    
    # Calculate coherence only for valid pixels
    if np.any(mask_valid):
        numerator = np.sqrt(diff[mask_valid]**2 + 4 * gxy_smooth[mask_valid]**2)
        denominator = trace[mask_valid]
        coherence_map[mask_valid] = numerator / denominator

    # --- 3. Mask Handling and Aggregation ---
    # Determine which mask to use for cell identification
    analysis_mask = None
    
    # Try to use provided segmentation masks
    if segmentation_masks and len(segmentation_masks) > 0:
        # Check the first mask (assuming it's the primary segmentation)
        candidate = segmentation_masks[0]
        if candidate is not None:
            # Ensure shape match
            if candidate.shape == tubulin.shape:
                analysis_mask = candidate
            elif candidate.ndim == 2 and tubulin.ndim == 2:
                 if candidate.shape == tubulin.shape:
                     analysis_mask = candidate

    # If no valid mask provided, generate one from the tubulin channel
    # This ensures we measure alignment within the biological structure
    if analysis_mask is None:
        try:
            thresh = threshold_otsu(tubulin)
            binary_mask = tubulin > thresh
            analysis_mask = label(binary_mask)
        except:
            # Fallback if otsu fails (e.g. uniform image)
            return 0.0

    # Ensure mask is integer labeled for regionprops
    if analysis_mask.dtype == bool:
        analysis_mask = label(analysis_mask)
    else:
        analysis_mask = analysis_mask.astype(int)

    # Compute mean coherence per cell
    # We calculate the mean coherence *within* each cell, then average across cells
    props = regionprops(analysis_mask, intensity_image=coherence_map)
    
    cell_scores = []
    for prop in props:
        # Filter tiny specks that are likely noise
        if prop.area > 50:
            cell_scores.append(prop.mean_intensity)
            
    if not cell_scores:
        return 0.0
        
    # Return the mean of the per-cell mean coherence
    return float(np.mean(cell_scores))
