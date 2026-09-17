def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.filters.rank import entropy
    from skimage.morphology import disk
    from skimage.filters import threshold_otsu
    
    # 1. Input Validation and Channel Extraction
    # Convert to numpy array if not already
    arr = np.asarray(img)
    
    # Check dimensionality and extract Actin channel (Channel 0)
    # Expected shape is (512, 512, 3)
    if arr.ndim == 3 and arr.shape[2] >= 1:
        actin_channel = arr[..., 0]  # Channel 0 is Actin
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on specs but safe)
        actin_channel = arr
    else:
        return 0.0

    # 2. Data Type Handling
    # skimage.filters.rank.entropy requires integer input (uint8 or uint16)
    # The dataset is uint8, but we ensure it here.
    if actin_channel.dtype != np.uint8:
        # Normalize to 0-255 and cast if not uint8
        if actin_channel.max() <= 1.0:
            actin_channel = (actin_channel * 255).astype(np.uint8)
        else:
            # Clip and cast
            actin_channel = np.clip(actin_channel, 0, 255).astype(np.uint8)

    # 3. Mask Generation (ROI Selection)
    # We only want to compute texture statistics within the biological cells,
    # not the background, to avoid skewing the entropy with large areas of zeros.
    
    mask = None
    
    # Check if segmentation masks are provided via *segmentation_masks
    if len(segmentation_masks) > 0:
        # Use the first available mask (usually whole cell or nuclei)
        # Assuming mask matches image spatial dimensions
        input_mask = segmentation_masks[0]
        if input_mask.shape == actin_channel.shape:
            mask = input_mask > 0
        elif input_mask.ndim == 3 and input_mask.shape[:2] == actin_channel.shape:
             # Handle case where mask might be 3D (e.g. labeled stack), flatten it
            mask = np.max(input_mask, axis=2) > 0
            
    # Fallback: Create a mask if none provided or valid
    if mask is None:
        # Check if image has content
        if actin_channel.max() == 0:
            return 0.0
        try:
            thresh = threshold_otsu(actin_channel)
            mask = actin_channel > thresh
        except Exception:
            # Fallback for extremely low contrast or empty images
            return 0.0

    # Ensure mask is boolean and has valid pixels
    if np.sum(mask) == 0:
        return 0.0

    # 4. Compute Local Entropy
    # We use a disk footprint. Radius 3 is a standard choice for cellular texture 
    # at this resolution (512x512), capturing local variations without being too global.
    footprint = disk(3)
    
    # Compute entropy map. 
    # Note: rank.entropy returns a float array where values are the local entropy.
    # We pass the mask to the function so it only computes/considers the relevant area,
    # or we compute globally and index later. Passing mask to rank filters usually 
    # handles boundary conditions better, but simple indexing is robust.
    # Here we compute the full map and index, as it's safer across versions.
    entropy_map = entropy(actin_channel, footprint)

    # 5. Aggregate Statistics
    # Extract entropy values only from the masked region
    roi_entropy_values = entropy_map[mask]
    
    if roi_entropy_values.size == 0:
        return 0.0
        
    # Calculate the mean entropy of the texture within the cells
    # High mean entropy -> Disordered cytoskeleton
    # Low mean entropy -> Smooth or highly organized/uniform regions
    result = np.mean(roi_entropy_values)

    return float(result)
