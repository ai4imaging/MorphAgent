def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import gabor, threshold_otsu
    
    # Convert to appropriate array type and normalize to [0, 1]
    # Image is (512, 512, 3) uint8
    # Channel 0 is Actin (Red), which is our target for texture analysis
    
    # Safety check for input dimensions
    if img.ndim < 3 or img.shape[-1] < 1:
        return 0.0
        
    # Extract Actin channel (Channel 0)
    actin_channel = img[..., 0].astype(np.float32)
    
    # Normalize to [0, 1] for consistent filter response
    # Using robust min/max to avoid hot pixel artifacts
    p_min, p_max = np.percentile(actin_channel, (1, 99))
    if p_max > p_min:
        actin_norm = (actin_channel - p_min) / (p_max - p_min)
    else:
        if p_max > 0:
            actin_norm = actin_channel / p_max
        else:
            actin_norm = actin_channel # All zeros
            
    actin_norm = np.clip(actin_norm, 0.0, 1.0)

    # Define Region of Interest (ROI)
    # We only want to compute texture energy inside cells, not on the background.
    # Background noise can artificially lower the average energy.
    
    mask = None
    
    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Combine all available masks (logical OR)
        combined_mask = np.zeros(actin_norm.shape, dtype=bool)
        for m in segmentation_masks:
            if m is not None and m.shape == actin_norm.shape:
                combined_mask = combined_mask | (m > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # 2. Fallback: Generate mask via Otsu thresholding on Actin channel if no external mask provided
    if mask is None:
        try:
            # Check if image is not essentially empty
            if np.max(actin_norm) > 0.05: # Threshold to avoid noise on blank images
                thresh = threshold_otsu(actin_norm)
                mask = actin_norm > thresh
            else:
                # Image is too dark/empty
                return 0.0
        except Exception:
            # Fallback for extremely uniform images where Otsu fails
            return 0.0

    # If mask is still empty (e.g. no cells found), return 0
    if not np.any(mask):
        return 0.0

    # Gabor Filter Parameters
    # Frequency: 0.1 to 0.25 is typical for cellular structures in 512px images.
    # We choose a frequency that highlights stress fibers (fine lines).
    frequency = 0.2
    
    # Orientations: 0, 45, 90, 135 degrees (0, pi/4, pi/2, 3pi/4 radians)
    # We want to detect fibers regardless of orientation.
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    # Accumulate maximum response across orientations
    # We take the max because a fiber oriented at 0 deg will have high response 
    # at 0 deg filter but low at 90 deg. The "energy" of that structure is best 
    # represented by its maximal response to the filter bank.
    max_response_map = np.zeros_like(actin_norm)
    
    for theta in thetas:
        # skimage.filters.gabor returns (real, imag)
        filt_real, filt_imag = gabor(actin_norm, frequency=frequency, theta=theta)
        
        # Compute magnitude (energy) for this orientation
        magnitude = np.sqrt(filt_real**2 + filt_imag**2)
        
        # Update the pixel-wise maximum
        max_response_map = np.maximum(max_response_map, magnitude)

    # Extract values only from the cellular regions
    roi_energies = max_response_map[mask]
    
    # Compute the mean energy
    # This quantifies the average strength of linear textures within the cell
    result = np.mean(roi_energies)

    return float(result)
