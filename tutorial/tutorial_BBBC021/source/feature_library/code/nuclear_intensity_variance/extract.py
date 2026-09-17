def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) where Channel 2 is DAPI (Nucleus)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract DAPI channel (Channel index 2)
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on spec, but safe)
        dapi_channel = arr
    else:
        return 0.0

    # Intensity normalization (0-255 -> 0.0-1.0)
    # This is critical for variance to be scale-independent relative to bit depth
    dapi_channel = dapi_channel / 255.0
    dapi_channel = np.clip(dapi_channel, 0.0, 1.0)

    # Handle segmentation masks
    labels = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask.shape == dapi_channel.shape:
            # If mask is boolean or binary (0/1), label it to get individual instances
            if mask.max() <= 1:
                labels = label(mask > 0)
            else:
                # Assume it's already an instance segmentation (integer labels)
                labels = mask.astype(int)
    
    # Fallback: Generate mask if none provided or invalid
    if labels is None:
        # Simple background subtraction/thresholding for fallback
        # Gaussian blur to reduce noise before thresholding
        blurred = ndimage.gaussian_filter(dapi_channel, sigma=2.0)
        
        # Check if image has content
        if np.max(blurred) > 0:
            try:
                thresh = threshold_otsu(blurred)
                binary_mask = blurred > thresh
                labels = label(binary_mask)
            except Exception:
                # Fallback for completely uniform images where otsu fails
                return 0.0
        else:
            return 0.0

    # Feature Computation: Nuclear Intensity Variance
    # We calculate the variance of pixel intensities *within* each nucleus, 
    # then average these variances across all nuclei in the image.
    
    # Get unique labels (excluding background 0)
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1: # Only background exists
        return 0.0
    
    # Use scipy.ndimage to efficiently calculate variance per labeled region
    # index=unique_labels[1:] skips the background (0)
    variances = ndimage.variance(dapi_channel, labels=labels, index=unique_labels[1:])
    
    # Filter out NaNs if any (e.g., single-pixel regions might have undefined variance depending on implementation, 
    # though ndimage usually handles this returning 0)
    variances = variances[~np.isnan(variances)]
    
    if len(variances) == 0:
        return 0.0

    # Return the mean of the variances
    # This represents the average "texture heterogeneity" of nuclei in the image
    result = np.mean(variances)

    return float(result)
