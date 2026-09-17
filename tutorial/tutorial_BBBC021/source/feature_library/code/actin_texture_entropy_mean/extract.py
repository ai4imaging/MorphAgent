def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters.rank import entropy
    from skimage.morphology import disk
    from skimage.filters import threshold_otsu
    from skimage.util import img_as_ubyte
    
    # 1. Input Validation and Preparation
    # Ensure input is an array
    arr = np.asarray(img)
    
    # Check dimensionality and extract Actin channel (Channel 0)
    # Dataset format: (512, 512, 3), Channel 0 = Actin
    if arr.ndim == 3 and arr.shape[2] >= 1:
        actin_channel = arr[:, :, 0]
    elif arr.ndim == 2:
        # Fallback if passed a single channel image, assume it's the relevant one
        actin_channel = arr
    else:
        return 0.0

    # 2. Ensure Data Type is uint8
    # Local entropy calculation relies on discrete bins (histograms). 
    # skimage.filters.rank.entropy requires uint8 input.
    if actin_channel.dtype != np.uint8:
        # If float, normalize to 0-1 then convert to uint8
        if np.issubdtype(actin_channel.dtype, np.floating):
            # Robust min-max normalization to preserve relative texture
            min_val, max_val = np.min(actin_channel), np.max(actin_channel)
            if max_val > min_val:
                actin_channel = (actin_channel - min_val) / (max_val - min_val)
            else:
                actin_channel = np.zeros_like(actin_channel)
            actin_channel = img_as_ubyte(actin_channel)
        else:
            # If integer but not uint8 (e.g. uint16), rescale or clip
            # Simple clipping for safety if range is unknown, but usually uint16 needs rescaling
            # Given dataset is uint8, this branch is just a safeguard
            actin_channel = actin_channel.astype(np.uint8)

    # 3. Define Region of Interest (ROI) / Mask
    # We want to compute texture entropy only within the cells/cytoplasm.
    foreground_mask = None

    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Combine all available masks to get a general "biological material" mask
        # Masks are typically labeled integers (0=bg, 1..N=cells)
        combined_mask = np.zeros(actin_channel.shape, dtype=bool)
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask shape matches image shape (handle potential mismatches)
                if mask.shape == actin_channel.shape:
                    combined_mask = combined_mask | (mask > 0)
        
        if np.any(combined_mask):
            foreground_mask = combined_mask

    # Fallback: If no masks provided or masks were empty, use Otsu thresholding on Actin
    if foreground_mask is None:
        try:
            thresh = threshold_otsu(actin_channel)
            foreground_mask = actin_channel > thresh
        except Exception:
            # Fallback for completely uniform images where Otsu fails
            return 0.0

    # If mask is still empty (e.g., black image), return 0
    if not np.any(foreground_mask):
        return 0.0

    # 4. Compute Local Entropy
    # Define neighborhood: A disk of radius 3 is standard for capturing local texture 
    # in 512x512 microscopy images without over-smoothing.
    footprint = disk(3)
    
    # Compute entropy map
    # The 'mask' parameter in rank filters ensures we only compute/consider values within the mask
    # However, passing 'mask' to rank.entropy means the neighborhood is restricted to the mask.
    # To get the entropy of the texture *at* the mask locations, we compute it.
    # We cast mask to uint8 as expected by some versions of skimage, though bool often works.
    try:
        entropy_map = entropy(actin_channel, footprint, mask=foreground_mask)
    except ValueError:
        # Handle edge cases where mask might be too small for the footprint
        return 0.0

    # 5. Aggregate Statistics
    # We only care about the entropy values inside the cell regions
    valid_entropy_values = entropy_map[foreground_mask]
    
    if valid_entropy_values.size == 0:
        return 0.0
        
    result = np.mean(valid_entropy_values)

    return float(result)
