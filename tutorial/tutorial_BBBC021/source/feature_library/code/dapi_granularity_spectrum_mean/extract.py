def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # Fallback for unexpected shapes, though dataset guarantees (512, 512, 3)
        return 0.0

    # Extract DAPI channel (Channel 2 - Blue)
    # Channel 0: Actin (Red), Channel 1: Tubulin (Green), Channel 2: DAPI (Blue)
    dapi_channel = arr[:, :, 2]

    # Intensity normalization
    # Normalize to [0, 1] based on robust range to ensure consistent spectral magnitude
    # Using 99.9 percentile to avoid hot pixel artifacts
    p_max = np.percentile(dapi_channel, 99.9)
    if p_max > 0:
        dapi_norm = dapi_channel / p_max
    else:
        dapi_norm = dapi_channel  # Should be all zeros
    
    dapi_norm = np.clip(dapi_norm, 0.0, 1.0)

    # Handle segmentation masks
    # We need a mask to isolate the nuclei. The feature is "dapi_granularity", 
    # so we must measure this only within the nuclear regions to avoid background noise.
    roi_mask = None

    # Check if any segmentation masks were provided
    if len(segmentation_masks) > 0:
        # Usually, if multiple masks are provided, one corresponds to nuclei.
        # Without explicit metadata on which is which in the tuple, we can try to infer 
        # or simply use the union of all masks if they represent cells.
        # However, typically the first mask is often the primary object (nuclei) or the masks are passed in order.
        # Let's assume the first mask available is usable for ROI definition.
        # If the mask is labeled (int), convert to binary.
        candidate_mask = segmentation_masks[0]
        if candidate_mask.shape == dapi_channel.shape:
            roi_mask = candidate_mask > 0
    
    # Fallback: Generate mask if none provided or invalid
    if roi_mask is None:
        # Simple background segmentation using Otsu
        # Add a small epsilon to avoid errors on completely black images
        if np.max(dapi_norm) > 0.05: # Threshold to ensure there's signal
            try:
                thresh = threshold_otsu(dapi_norm)
                roi_mask = dapi_norm > thresh
            except ValueError:
                # Can happen if image is uniform
                roi_mask = np.zeros_like(dapi_norm, dtype=bool)
        else:
            roi_mask = np.zeros_like(dapi_norm, dtype=bool)

    # Feature Computation: High-Frequency Power Spectrum Proxy
    # Instead of a global FFT which introduces windowing artifacts due to non-rectangular ROIs (cells),
    # we use the Laplacian filter in the spatial domain.
    # By Parseval's theorem, the energy (sum of squares) in the frequency domain is related to energy in spatial domain.
    # The Laplacian is a high-pass filter. Its response magnitude corresponds to high-frequency energy (edges, texture).
    
    # Apply Laplacian filter to detect high-frequency variations (granularity)
    # We use a standard discrete Laplacian kernel
    laplacian_response = ndimage.laplace(dapi_norm)
    
    # Compute the "Power" (squared magnitude) of the high-frequency components
    power_map = laplacian_response ** 2

    # Extract values only within the ROI (Nuclei)
    # If the mask is empty, return 0.0
    if np.sum(roi_mask) == 0:
        return 0.0
    
    # Get the power values inside the nuclei
    roi_power = power_map[roi_mask]
    
    # Compute the mean power
    # This represents the average intensity of high-frequency components (granularity) within the nuclei.
    result = np.mean(roi_power)

    return float(result)
