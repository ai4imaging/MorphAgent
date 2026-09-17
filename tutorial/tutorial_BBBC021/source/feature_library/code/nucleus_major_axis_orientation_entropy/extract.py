def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from scipy import stats
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected 3-channel image, return 0.0
        return 0.0

    # Identify the Nucleus Channel (Channel 2 = Blue = DAPI)
    nucleus_channel = arr[:, :, 2]

    # Normalize intensity for processing
    vmax = np.percentile(nucleus_channel, 99.5) if nucleus_channel.size > 0 else 1.0
    if vmax > 0:
        nucleus_channel = nucleus_channel / vmax
    nucleus_channel = np.clip(nucleus_channel, 0.0, 1.0)

    # Determine Segmentation Mask
    # We prefer a provided segmentation mask if available.
    # Assuming the first mask in the list is likely the nuclear mask or a general cell mask.
    labeled_mask = None
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the provided mask
        mask_input = segmentation_masks[0]
        # Ensure it matches image dimensions (2D)
        if mask_input.shape == nucleus_channel.shape:
            # If it's already labeled (int), use it. If binary, label it.
            if np.issubdtype(mask_input.dtype, np.integer) and np.max(mask_input) > 1:
                labeled_mask = mask_input
            else:
                labeled_mask = label(mask_input > 0)
    
    # Fallback: Generate mask from DAPI channel if no valid mask provided
    if labeled_mask is None:
        # Simple thresholding pipeline
        try:
            thresh = threshold_otsu(nucleus_channel)
            binary_mask = nucleus_channel > thresh
            # Clean up noise
            binary_mask = binary_opening(binary_mask, disk(2))
            labeled_mask = label(binary_mask)
        except Exception:
            # Fallback if thresholding fails (e.g., empty image)
            return 0.0

    # Extract Region Properties
    # We need 'orientation' from regionprops.
    # orientation: Angle between the 0th axis (rows) and the major axis of the ellipse 
    # that has the same second-moments as the region. Range is [-pi/2, pi/2].
    props = regionprops(labeled_mask)

    # Collect orientations
    orientations = []
    for prop in props:
        # Filter out very small artifacts to reduce noise
        if prop.area > 50:
            orientations.append(prop.orientation)

    # Calculate Entropy
    if len(orientations) < 2:
        # If 0 or 1 nucleus, entropy is 0 (no disorder/distribution to measure)
        return 0.0

    # Create a histogram of orientations
    # Range is -pi/2 to pi/2 (approx -1.57 to 1.57). Total span is pi.
    # We use 18 bins, so each bin covers 10 degrees (pi/18 radians).
    hist, bin_edges = np.histogram(orientations, bins=18, range=(-np.pi/2, np.pi/2), density=False)

    # Normalize to get probabilities
    total_nuclei = np.sum(hist)
    if total_nuclei == 0:
        return 0.0
    
    probabilities = hist / total_nuclei

    # Calculate Shannon Entropy
    # We use base 2 for bits. High entropy = random orientation. Low entropy = aligned.
    # scipy.stats.entropy uses natural log by default, so we specify base=2
    entropy_val = stats.entropy(probabilities, base=2)

    return float(entropy_val)
