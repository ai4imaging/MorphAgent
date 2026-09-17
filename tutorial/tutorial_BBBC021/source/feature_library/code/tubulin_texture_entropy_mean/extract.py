def extract(img, *segmentation_masks):
    import numpy as np
    from skimage.filters.rank import entropy
    from skimage.morphology import disk
    from skimage.filters import threshold_otsu
    from skimage.util import img_as_ubyte

    # 1. Check input validity
    if img is None or img.size == 0:
        return 0.0

    # 2. Extract Tubulin Channel (Channel 1 - Green) based on dataset description
    # Dimensions are (Height, Width, Channels) = (512, 512, 3)
    try:
        if img.ndim == 3 and img.shape[2] >= 2:
            tubulin_channel = img[:, :, 1]
        elif img.ndim == 2:
            # Fallback if passed a single channel image, though unlikely given description
            tubulin_channel = img
        else:
            return 0.0
    except Exception:
        return 0.0

    # 3. Ensure image is uint8 for rank filters
    # The dataset description says uint8, but we ensure it for safety
    if tubulin_channel.dtype != np.uint8:
        # Normalize to 0-255 if not already
        if tubulin_channel.max() <= 1.0:
            tubulin_channel = (tubulin_channel * 255).astype(np.uint8)
        else:
            # Clip and cast
            tubulin_channel = np.clip(tubulin_channel, 0, 255).astype(np.uint8)

    # 4. Determine Region of Interest (ROI)
    # We want to calculate entropy only within the cell bodies to avoid background noise
    mask = None
    
    if len(segmentation_masks) > 0:
        # Use provided segmentation masks
        # Combine all masks to get the total cellular area
        combined_mask = np.zeros(tubulin_channel.shape, dtype=bool)
        for seg in segmentation_masks:
            if seg is not None and seg.shape == tubulin_channel.shape:
                combined_mask = np.logical_or(combined_mask, seg > 0)
        
        if np.any(combined_mask):
            mask = combined_mask

    if mask is None:
        # Fallback: Generate a mask using Otsu thresholding on the tubulin channel
        # This separates the fluorescent structure from the dark background
        try:
            thresh = threshold_otsu(tubulin_channel)
            mask = tubulin_channel > thresh
        except Exception:
            # If thresholding fails (e.g. uniform image), use whole image
            mask = np.ones(tubulin_channel.shape, dtype=bool)

    # Check if mask is empty
    if not np.any(mask):
        return 0.0

    # 5. Compute Local Entropy
    # We use a disk footprint to define the local neighborhood.
    # Radius 4 is chosen for 512x512 images to capture texture scale of microtubules.
    # entropy() returns the entropy of the local neighborhood for each pixel.
    try:
        footprint = disk(4)
        # We apply the mask *after* calculation or use the mask parameter if supported by implementation,
        # but skimage.filters.rank.entropy calculates for the whole image usually.
        # Calculating on the whole image and then masking is safer for boundary conditions.
        entropy_map = entropy(tubulin_channel, footprint)
    except Exception:
        return 0.0

    # 6. Aggregate Statistics
    # Extract entropy values only from the masked region (the cells)
    masked_entropy = entropy_map[mask]

    if masked_entropy.size == 0:
        return 0.0

    # Calculate the mean entropy
    mean_entropy = np.mean(masked_entropy)

    return float(mean_entropy)
