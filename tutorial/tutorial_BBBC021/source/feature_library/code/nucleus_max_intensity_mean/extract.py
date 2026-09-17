def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects

    # 1. Channel Selection
    # Dataset description: Channel 2 is DAPI (Nucleus).
    # Image shape is (512, 512, 3).
    try:
        if img.ndim == 3 and img.shape[2] >= 3:
            # Extract Blue channel (Index 2)
            nucleus_channel = img[:, :, 2]
        elif img.ndim == 2:
            # Fallback if passed a single channel image (unlikely given spec but safe)
            nucleus_channel = img
        else:
            return 0.0
    except Exception:
        return 0.0

    # Convert to float for processing, though max intensity is conceptually based on raw values.
    # We keep the raw values for the measurement to be meaningful in the context of 0-255 uint8 data,
    # but we cast to float to avoid overflow during mean calculation if needed.
    # However, for "max intensity", we usually want the value in the original range (0-255) or normalized range.
    # Given the prompt asks for "intensity", preserving the relative scale is key.
    # Let's work with the raw values but return a float.
    nucleus_channel_float = nucleus_channel.astype(np.float32)

    # 2. Mask Resolution
    # Check if segmentation masks are provided
    labeled_mask = None
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Assume the first mask is the relevant one (nuclei/cells)
        mask_input = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[-2:] == nucleus_channel.shape[-2:]:
            # If mask is already labeled (int), use it. If binary (bool/0-1), label it.
            if np.issubdtype(mask_input.dtype, np.integer) and mask_input.max() > 1:
                labeled_mask = mask_input
            else:
                labeled_mask = label(mask_input > 0)
        else:
            # Shape mismatch, ignore provided mask
            labeled_mask = None

    # Fallback Segmentation if no valid mask provided
    if labeled_mask is None:
        try:
            # Simple Otsu thresholding on the nucleus channel
            # Smooth slightly to reduce noise
            smoothed = ndimage.gaussian_filter(nucleus_channel_float, sigma=2)
            thresh = threshold_otsu(smoothed)
            binary_mask = smoothed > thresh
            
            # Remove small artifacts (noise)
            binary_mask = remove_small_objects(binary_mask, min_size=50)
            
            # Label the regions
            labeled_mask = label(binary_mask)
        except Exception:
            # If segmentation fails (e.g., empty image), return 0.0
            return 0.0

    # 3. Feature Computation
    # Get unique labels (excluding background 0)
    # We can use ndimage.find_objects or simply pass index to ndimage functions
    # ndimage.maximum is efficient for this.
    
    # Get all unique labels in the mask
    unique_labels = np.unique(labeled_mask)
    # Remove background (0)
    unique_labels = unique_labels[unique_labels != 0]

    if len(unique_labels) == 0:
        return 0.0

    # Calculate the maximum intensity within each labeled region
    # index=unique_labels ensures we get a list corresponding to each cell
    max_intensities = ndimage.maximum(nucleus_channel_float, labels=labeled_mask, index=unique_labels)

    # 4. Aggregation
    # Compute the mean of these maximums
    if max_intensities.size == 0:
        return 0.0
        
    result = np.mean(max_intensities)

    return float(result)
