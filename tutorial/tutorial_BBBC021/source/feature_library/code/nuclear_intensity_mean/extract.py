def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters import threshold_otsu, gaussian
    
    # 1. Data Validation and Preprocessing
    # Convert to float32 for calculations
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensionality based on dataset description (512, 512, 3)
    # Return 0.0 if dimensions are unexpected
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # 2. Channel Selection
    # According to dataset info: Channel 2 (Index 2) is DAPI (Nucleus)
    dapi_channel = arr[:, :, 2]

    # 3. Normalization
    # Data is uint8 (0-255). Normalize to [0, 1] range.
    # We use fixed scaling because intensity is a quantitative measure here.
    dapi_norm = dapi_channel / 255.0
    
    # Clip to ensure range [0, 1]
    dapi_norm = np.clip(dapi_norm, 0.0, 1.0)

    # 4. Mask Generation / Selection
    binary_mask = None

    # Scenario A: Segmentation masks provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches spatial dimensions of the image
        if mask_input.shape == dapi_norm.shape:
            # Create boolean mask (foreground > 0)
            binary_mask = mask_input > 0
        else:
            # If dimensions mismatch, fallback to auto-segmentation
            pass

    # Scenario B: No mask provided or invalid mask (Fallback)
    if binary_mask is None:
        # Apply slight smoothing to reduce noise before thresholding
        blurred = gaussian(dapi_norm, sigma=1)
        
        # Calculate Otsu threshold
        # Handle case where image is uniform (e.g., all black)
        if np.min(blurred) == np.max(blurred):
            thresh = 0
        else:
            thresh = threshold_otsu(blurred)
            
        binary_mask = dapi_norm > thresh

    # 5. Feature Computation
    # Select pixels belonging to nuclei
    nuclear_pixels = dapi_norm[binary_mask]

    # Handle edge case: No nuclear pixels found
    if nuclear_pixels.size == 0:
        return 0.0

    # Compute mean intensity
    mean_intensity = np.mean(nuclear_pixels)

    return float(mean_intensity)
