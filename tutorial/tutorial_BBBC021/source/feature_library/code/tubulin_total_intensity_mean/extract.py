def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label

    # 1. Data Preparation
    # Ensure image is valid
    if img is None or img.size == 0:
        return 0.0

    # Handle dimensionality
    # Dataset description: (512, 512, 3) where Channel 1 is Tubulin (Green)
    # Check if we have at least 3 dimensions and the last dimension is channels
    if img.ndim == 3 and img.shape[2] >= 2:
        # Extract Tubulin channel (Channel 1)
        tubulin_channel = img[:, :, 1]
        # For fallback segmentation, we might use Actin (Channel 0) if available, or Tubulin
        actin_channel = img[:, :, 0]
    elif img.ndim == 2:
        # If passed a single channel 2D image, assume it's the relevant channel
        tubulin_channel = img
        actin_channel = img
    else:
        # Unexpected format
        return 0.0

    # Convert to float64 for precise summation (avoid uint8 overflow)
    tubulin_channel = tubulin_channel.astype(np.float64)

    # 2. Segmentation Handling
    labeled_mask = None

    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Heuristic: Select the mask with the largest foreground area.
        # This usually prioritizes whole-cell masks over nuclear masks, which is better for Tubulin.
        best_mask = None
        max_area = -1

        for mask in segmentation_masks:
            if mask is None:
                continue
            
            # Ensure mask is 2D (match image spatial dims)
            if mask.ndim > 2:
                mask = np.max(mask, axis=0) # Simple projection if 3D mask passed
            
            if mask.shape != tubulin_channel.shape:
                continue

            current_area = np.count_nonzero(mask)
            if current_area > max_area:
                max_area = current_area
                best_mask = mask
        
        if best_mask is not None:
            # Ensure it's labeled (integers) not just binary
            if best_mask.dtype == bool or np.max(best_mask) == 1:
                labeled_mask = label(best_mask)
            else:
                labeled_mask = best_mask

    # Fallback: Generate segmentation on the fly if no masks provided
    if labeled_mask is None:
        # Use Actin channel for cell shape if available, else Tubulin
        # Actin (Channel 0) usually defines cell boundaries better than Tubulin
        segmentation_source = actin_channel if 'actin_channel' in locals() else tubulin_channel
        
        # Smooth to reduce noise
        smoothed = ndimage.gaussian_filter(segmentation_source, sigma=2)
        
        # Threshold
        try:
            thresh = threshold_otsu(smoothed)
            binary_mask = smoothed > thresh
        except Exception:
            # Fallback for extremely low contrast or empty images
            binary_mask = smoothed > np.mean(smoothed)
            
        # Label connected components
        labeled_mask = label(binary_mask)

    # 3. Feature Computation
    # Get unique labels (excluding background 0)
    unique_labels = np.unique(labeled_mask)
    if unique_labels.size > 0 and unique_labels[0] == 0:
        unique_labels = unique_labels[1:]

    if unique_labels.size == 0:
        return 0.0

    # Calculate Sum of Intensity per object
    # scipy.ndimage.sum is efficient for this
    # It returns a list of sums corresponding to the index list provided
    total_intensities = ndimage.sum(input=tubulin_channel, labels=labeled_mask, index=unique_labels)

    # 4. Aggregation
    # Calculate the mean of the total intensities
    mean_total_intensity = np.mean(total_intensities)

    return float(mean_total_intensity)
