def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label

    # 1. Input Validation and Preprocessing
    # Ensure image is valid
    if img is None or img.size == 0:
        return 0.0

    # Handle dimensionality
    # Dataset is (512, 512, 3). Channel 2 is DAPI (Nucleus).
    # If the image is 2D (H, W), assume it's a single channel or projection.
    # If 3D (H, W, C), extract the nuclear channel.
    
    nuclear_channel = None
    
    if img.ndim == 3:
        if img.shape[-1] == 3: # (H, W, 3)
            # Channel 2 is Blue/DAPI based on dataset info
            nuclear_channel = img[..., 2]
        else:
            # Fallback: take the last channel or max projection if unknown structure
            nuclear_channel = img[..., -1]
    elif img.ndim == 2:
        nuclear_channel = img
    else:
        return 0.0

    # Convert to float64 to prevent overflow during summation of intensities
    # uint8 sums can easily exceed 255 or 65535
    nuclear_channel = nuclear_channel.astype(np.float64)

    # 2. Determine Segmentation Mask
    # We need a labeled mask where each nucleus has a unique integer ID.
    labeled_mask = None

    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first mask, assuming it corresponds to nuclei (common convention)
        # or specifically the one matching the nuclear channel.
        # Based on typical pipelines, mask 0 is often nuclei or cells.
        # We will assume mask 0 is the relevant one for nuclei.
        mask_input = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == nuclear_channel.shape[:2]:
            # If the mask is already labeled (max > 1), use it directly
            if mask_input.max() > 1:
                labeled_mask = mask_input.astype(np.int32)
            else:
                # If binary (0/1), label connected components
                labeled_mask = label(mask_input > 0)
    
    # Fallback: Generate mask if none provided
    if labeled_mask is None:
        # Simple Otsu thresholding on the nuclear channel
        # Smooth slightly to reduce noise
        smoothed = ndimage.gaussian_filter(nuclear_channel, sigma=2)
        try:
            thresh = threshold_otsu(smoothed)
            binary_mask = smoothed > thresh
            labeled_mask = label(binary_mask)
        except Exception:
            # Fallback if otsu fails (e.g. constant image)
            return 0.0

    # 3. Compute Feature: Integrated Intensity per Nucleus
    # Get unique labels, excluding background (0)
    # Using ndimage.find_objects or unique is safer, but for sum we can just pass index
    # However, we need to know which indices exist to compute std dev correctly.
    
    # Get unique labels present in the mask (excluding 0)
    unique_labels = np.unique(labeled_mask)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) < 2:
        # Standard deviation requires at least two data points to be meaningful for population heterogeneity,
        # though numpy returns 0.0 for 1 item.
        # If 0 items, return 0.0.
        if len(unique_labels) == 1:
            return 0.0 
        return 0.0

    # Calculate sum of pixel intensities for each labeled region
    # ndimage.sum returns a list of sums corresponding to the indices provided
    integrated_intensities = ndimage.sum(nuclear_channel, labeled_mask, index=unique_labels)

    # 4. Calculate Standard Deviation
    # We want the standard deviation of these integrated intensity values
    # ddof=1 (sample std) is usually preferred for statistical estimation, 
    # but ddof=0 is standard for descriptive image features. We'll use ddof=1 for "heterogeneity".
    # If N < 2, we already returned 0.0.
    
    std_val = np.std(integrated_intensities, ddof=1)

    return float(std_val)
