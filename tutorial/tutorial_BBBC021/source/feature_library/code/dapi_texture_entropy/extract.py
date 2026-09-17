def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import stats
    from skimage.filters import threshold_otsu
    from skimage.measure import label
    
    # 1. Input Validation and Channel Extraction
    # Convert to numpy array if not already
    img = np.asarray(img)
    
    # Handle dimensionality
    # Expected shape is (512, 512, 3) for the composite image
    # Channel 2 (index 2) is DAPI (Blue)
    if img.ndim == 3 and img.shape[2] >= 3:
        dapi_channel = img[:, :, 2]
    elif img.ndim == 2:
        # Fallback: if a single channel is passed, assume it's the relevant one
        dapi_channel = img
    else:
        # Unexpected format
        return 0.0

    # Ensure dapi_channel is uint8 for consistent histogram binning (0-255)
    # If it's float, normalize and convert. If it's integer but not uint8, clip and cast.
    if np.issubdtype(dapi_channel.dtype, np.floating):
        # Normalize float [0, 1] to [0, 255]
        dapi_channel = np.clip(dapi_channel * 255, 0, 255).astype(np.uint8)
    elif dapi_channel.dtype != np.uint8:
        # Clip any other integer types to valid range
        dapi_channel = np.clip(dapi_channel, 0, 255).astype(np.uint8)

    # 2. Mask Handling
    # We need a labeled mask to identify individual nuclei.
    # If a segmentation mask is provided, use it. Otherwise, generate one.
    labeled_mask = None
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == dapi_channel.shape[:2]:
            # If mask is boolean or binary (0/1), label it to separate objects
            if mask_input.dtype == bool or mask_input.max() <= 1:
                labeled_mask = label(mask_input)
            else:
                # Assume it's already an integer label map
                labeled_mask = mask_input.astype(int)
    
    # Fallback: Generate mask if none provided or invalid
    if labeled_mask is None:
        try:
            # Simple Otsu thresholding to separate foreground (nuclei) from background
            thresh = threshold_otsu(dapi_channel)
            binary_mask = dapi_channel > thresh
            labeled_mask = label(binary_mask)
        except Exception:
            # If thresholding fails (e.g., uniform image), return 0
            return 0.0

    # 3. Feature Computation: Entropy per Nucleus
    # We compute entropy for each nucleus individually to avoid inter-cell variance affecting the score.
    # Then we average the results.
    
    # Get unique labels (excluding 0 which is background)
    unique_labels = np.unique(labeled_mask)
    if len(unique_labels) <= 1: # Only background exists
        return 0.0
    
    entropies = []
    
    # Iterate over each cell
    for region_id in unique_labels:
        if region_id == 0:
            continue
            
        # Extract pixels for this specific nucleus
        # Using boolean indexing
        nucleus_pixels = dapi_channel[labeled_mask == region_id]
        
        if nucleus_pixels.size == 0:
            continue
            
        # Compute histogram of pixel intensities
        # We use fixed bins 0-256 for uint8 data
        counts, _ = np.histogram(nucleus_pixels, bins=256, range=(0, 256))
        
        # Normalize counts to get probabilities
        # Avoid division by zero
        total_pixels = counts.sum()
        if total_pixels > 0:
            probs = counts / total_pixels
            # Compute Shannon entropy (base 2)
            # scipy.stats.entropy uses base e by default, so we specify base=2
            ent = stats.entropy(probs, base=2)
            entropies.append(ent)
            
    # 4. Aggregation
    if not entropies:
        return 0.0
        
    # Return the mean entropy across all detected nuclei
    result = np.mean(entropies)
    
    return float(result)
