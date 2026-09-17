def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects

    # Convert to appropriate array type
    # Image is (512, 512, 3), uint8
    # Channel 0 is Actin (Red)
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality and channel selection
    # We need the Actin channel (Channel 0) to measure cell spreading/area
    if arr.ndim == 3 and arr.shape[2] >= 1:
        actin_channel = arr[:, :, 0]
    elif arr.ndim == 2:
        # Fallback if single channel image is passed (unlikely based on description but safe)
        actin_channel = arr
    else:
        return 0.0

    # Determine the label image (segmentation)
    label_img = None

    # Strategy 1: Use provided segmentation mask if available
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == actin_channel.shape[:2]:
            # If the mask is already labeled (integer labels > 1), use it directly
            # If it's a binary mask (0 and 1), label it
            if mask_input.max() > 1:
                label_img = mask_input.astype(int)
            else:
                label_img = label(mask_input > 0)

    # Strategy 2: Compute segmentation on-the-fly using Actin channel
    if label_img is None:
        # Normalize Actin channel for segmentation
        # Robust min/max to handle outliers
        p_min, p_max = np.percentile(actin_channel, (1, 99))
        if p_max > p_min:
            norm_actin = (actin_channel - p_min) / (p_max - p_min)
        else:
            norm_actin = actin_channel
        norm_actin = np.clip(norm_actin, 0, 1)

        # Smooth to reduce noise and improve object connectivity
        # Sigma=2.0 is reasonable for 512x512 cell images to merge internal texture
        smoothed = ndimage.gaussian_filter(norm_actin, sigma=2.0)

        # Thresholding
        try:
            thresh = threshold_otsu(smoothed)
            binary_mask = smoothed > thresh
        except Exception:
            # Fallback if image is uniform (e.g., empty)
            return 0.0

        # Post-processing morphological cleanup
        # Fill holes to ensure the whole cell area is counted, not just the cytoskeleton ring
        binary_mask = ndimage.binary_fill_holes(binary_mask)
        
        # Remove small artifacts (debris)
        # 50 pixels is a conservative lower bound for a cell at this resolution
        binary_mask = remove_small_objects(binary_mask, min_size=50)

        # Label connected components
        label_img = label(binary_mask)

    # Compute Feature: Mean Area
    # regionprops returns a list of properties for each labeled region
    regions = regionprops(label_img)

    if not regions:
        return 0.0

    # Extract area for each cell
    areas = [r.area for r in regions]

    # Calculate mean
    mean_area = np.mean(areas)

    return float(mean_area)
