def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import gabor, threshold_otsu
    
    # 1. Data Validation and Preparation
    # Convert to float32 for processing
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensionality and extract Tubulin channel (Index 1)
    # Expected shape: (H, W, 3) -> (512, 512, 3)
    if arr.ndim == 3 and arr.shape[2] >= 2:
        # Extract Green/Tubulin channel
        tubulin = arr[..., 1]
    elif arr.ndim == 2:
        # Fallback if only 2D image is passed (unlikely based on spec, but safe)
        tubulin = arr
    else:
        return 0.0

    # Normalize intensity to [0, 1]
    # The input is uint8 (0-255), so we divide by 255.0
    # If the input was already float or had a different range, we ensure 0-1 range
    if np.max(tubulin) > 1.0:
        tubulin = tubulin / 255.0
    tubulin = np.clip(tubulin, 0.0, 1.0)

    # 2. Define Region of Interest (ROI)
    # We want to measure texture only within the cells, not the background.
    roi_mask = None

    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Combine all available masks
        combined_mask = np.zeros(tubulin.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None and mask.shape == tubulin.shape:
                combined_mask = np.logical_or(combined_mask, mask > 0)
        
        if np.any(combined_mask):
            roi_mask = combined_mask

    # Fallback: If no masks provided or masks are empty, generate a mask using Otsu
    if roi_mask is None:
        try:
            # Check if image is not empty/black
            if np.max(tubulin) > 0:
                thresh = threshold_otsu(tubulin)
                roi_mask = tubulin > thresh
            else:
                return 0.0
        except Exception:
            # Fallback for extremely low contrast or uniform images
            return 0.0

    # If mask is still empty (e.g., no cells detected), return 0
    if not np.any(roi_mask):
        return 0.0

    # 3. Gabor Filter Bank Configuration
    # Microtubules are fine, directional structures.
    # We use a specific frequency to target these filaments.
    # Frequency: 0.1 to 0.25 is typical for cellular textures at this resolution.
    frequency = 0.2
    
    # Orientations: 0, 45, 90, 135 degrees to capture fibers in all directions.
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    # Sigma: Bandwidth of the filter.
    sigma_x = 3.0
    sigma_y = 3.0

    # 4. Compute Gabor Energy
    # We sum the energy responses across all orientations to get a rotation-invariant measure.
    total_energy_map = np.zeros_like(tubulin)

    for theta in thetas:
        # skimage.filters.gabor returns real and imaginary parts
        filt_real, filt_imag = gabor(
            tubulin, 
            frequency=frequency, 
            theta=theta,
            sigma_x=sigma_x, 
            sigma_y=sigma_y
        )
        
        # Calculate magnitude (energy) for this orientation
        # Energy = sqrt(real^2 + imag^2)
        energy = np.hypot(filt_real, filt_imag)
        
        # Accumulate
        total_energy_map += energy

    # 5. Aggregate Statistics
    # Extract energy values only from the ROI (cellular regions)
    masked_energy = total_energy_map[roi_mask]
    
    # Compute the mean energy
    # High mean energy -> Strong, distinct microtubule fibers (bundling or healthy network)
    # Low mean energy -> Diffuse signal (depolymerization) or weak texture
    result = np.mean(masked_energy)

    return float(result)
