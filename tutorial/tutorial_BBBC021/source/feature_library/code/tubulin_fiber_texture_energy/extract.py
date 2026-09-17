def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) -> (Height, Width, Channels)
    # Channel 1 is Tubulin (Green)
    if arr.ndim == 3 and arr.shape[2] == 3:
        tubulin_channel = arr[..., 1]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on spec but safe)
        tubulin_channel = arr
    else:
        return 0.0

    # Determine Region of Interest (ROI)
    # We want to calculate texture energy only within the cellular regions,
    # not the background, to avoid diluting the signal.
    
    mask = None
    
    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0:
        # Combine all available masks (Union)
        combined_mask = np.zeros(tubulin_channel.shape, dtype=bool)
        for m in segmentation_masks:
            if m is not None:
                # Ensure mask matches image shape (handle potential 2D/3D mismatches)
                if m.shape == tubulin_channel.shape:
                    combined_mask = np.logical_or(combined_mask, m > 0)
                elif m.ndim == 3 and m.shape[:2] == tubulin_channel.shape:
                     # If mask is 3D (e.g. labeled), flatten it
                    combined_mask = np.logical_or(combined_mask, np.max(m, axis=2) > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    # 2. Fallback: If no valid masks provided, generate a foreground mask
    if mask is None:
        # Use Otsu thresholding on the tubulin channel itself to find cells
        # Check if image is not empty/constant
        if np.max(tubulin_channel) > np.min(tubulin_channel):
            thresh = threshold_otsu(tubulin_channel)
            mask = tubulin_channel > thresh
        else:
            # If image is constant (e.g. all black), return 0
            return 0.0

    # Ensure we have a valid mask with pixels
    if not np.any(mask):
        return 0.0

    # Compute Texture Energy
    # We use a Laplacian filter to detect high-frequency edges (fibers).
    # Diffuse tubulin (depolymerized) is smooth -> Low Laplacian response.
    # Bundled/Fibrous tubulin (stabilized) has sharp edges -> High Laplacian response.
    
    # Apply Laplacian filter
    # We do not normalize the image to 0-1 range before this, because the absolute 
    # intensity contrast of the fibers is relevant for "energy".
    laplacian_response = ndimage.laplace(tubulin_channel)
    
    # Calculate the "Energy"
    # Energy is typically defined as the mean squared or mean absolute value of the filter response.
    # We use Mean Absolute Deviation (L1 norm) as it is robust and standard for texture analysis.
    
    # Extract values only within the cellular mask
    roi_values = laplacian_response[mask]
    
    # Compute Mean Absolute Value
    texture_energy = np.mean(np.abs(roi_values))

    return float(texture_energy)
