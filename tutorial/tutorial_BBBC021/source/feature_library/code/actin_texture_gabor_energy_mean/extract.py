def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import gabor, threshold_otsu
    from skimage.morphology import binary_closing, disk
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) uint8. Channel 0 is Actin.
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract Actin channel (Channel 0)
        actin_channel = arr[:, :, 0]
    elif arr.ndim == 2:
        # Fallback if single channel passed (assume it's the relevant one)
        actin_channel = arr
    else:
        # Unexpected format
        return 0.0

    # Intensity normalization [0, 1]
    # uint8 range is 0-255
    actin_channel = actin_channel / 255.0
    
    # Clip to ensure bounds
    actin_channel = np.clip(actin_channel, 0.0, 1.0)

    # Define Region of Interest (ROI)
    # We want to calculate texture energy only within the cells, not the background.
    roi_mask = None

    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Combine all available masks to get a general "cellular area" mask
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask matches image dimensions (handle potential 3D vs 2D mismatch if any)
                if mask.shape == actin_channel.shape:
                    combined_mask = np.logical_or(combined_mask, mask > 0)
        
        if np.any(combined_mask):
            roi_mask = combined_mask

    # 2. Fallback: Generate mask from image if no valid segmentation provided
    if roi_mask is None:
        # Check if image has content
        if np.max(actin_channel) > 0.05: # Threshold to avoid noise in empty images
            try:
                thresh = threshold_otsu(actin_channel)
                roi_mask = actin_channel > thresh
                # Clean up the mask slightly
                roi_mask = binary_closing(roi_mask, disk(3))
            except Exception:
                # Fallback for extremely low contrast images where otsu might fail
                roi_mask = actin_channel > 0.1
        else:
            # Image is effectively empty
            return 0.0

    # Ensure we have a valid mask with pixels
    if roi_mask is None or np.sum(roi_mask) == 0:
        return 0.0

    # Gabor Filter Bank Configuration
    # We want to detect linear structures (stress fibers) at various orientations.
    # Frequency: 0.2 corresponds to a wavelength of 5 pixels, suitable for fibers.
    frequency = 0.2
    # Orientations: 0, 45, 90, 135 degrees
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    total_energy_sum = 0.0
    
    # Apply filter bank
    # We sum the energy responses across all orientations to get a rotation-invariant measure
    # of "fibrousness".
    accumulated_energy_map = np.zeros_like(actin_channel)
    
    for theta in thetas:
        # gabor returns real and imaginary parts
        filt_real, filt_imag = gabor(actin_channel, frequency=frequency, theta=theta, sigma_x=3, sigma_y=3)
        
        # Calculate magnitude (energy) for this orientation
        # Energy = sqrt(real^2 + imag^2)
        energy = np.hypot(filt_real, filt_imag)
        
        accumulated_energy_map += energy

    # Average the accumulated energy over the ROI (cellular area)
    # We use the mask to exclude background noise from the mean calculation
    roi_pixels = accumulated_energy_map[roi_mask]
    
    if roi_pixels.size == 0:
        return 0.0
        
    mean_energy = np.mean(roi_pixels)

    return float(mean_energy)
