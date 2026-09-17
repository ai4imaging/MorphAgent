def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    import math

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality
    # Expected: (H, W, C) where C >= 3. Channel 2 is Nucleus (Blue).
    if arr.ndim == 3 and arr.shape[2] >= 3:
        # Extract Blue channel (Nucleus)
        nuclei_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely given description but safe)
        nuclei_channel = arr
    else:
        return 0.0

    # Normalize intensity
    vmax = np.percentile(nuclei_channel, 99.5) if nuclei_channel.size > 0 else 1.0
    if vmax > 0:
        nuclei_channel = nuclei_channel / vmax
    nuclei_channel = np.clip(nuclei_channel, 0.0, 1.0)

    # Determine Segmentation Mask
    # We prioritize provided masks. If not available or empty, we compute one.
    final_mask = None
    
    if len(segmentation_masks) > 0:
        # Check if any mask is valid (non-empty)
        # We iterate through provided masks to find a suitable one (assuming one of them is nuclei)
        # Since the prompt doesn't specify which mask is which index, and often nuclei is primary,
        # we check for a mask that seems to segment objects.
        for mask in segmentation_masks:
            if mask is not None and np.any(mask):
                final_mask = mask
                break
    
    # Fallback: Compute mask using Otsu if no valid mask provided
    if final_mask is None:
        try:
            thresh = threshold_otsu(nuclei_channel)
            final_mask = nuclei_channel > thresh
        except Exception:
            return 0.0

    # Ensure mask is integer labeled
    if final_mask.dtype == bool:
        labeled_mask = label(final_mask)
    else:
        labeled_mask = final_mask.astype(int)

    # Compute Region Properties
    props = regionprops(labeled_mask)

    if not props:
        return 0.0

    roughness_scores = []

    for prop in props:
        # Filter small artifacts
        if prop.area < 50:
            continue

        # Roughness Metric: (Perimeter^2 / (4 * pi * Area)) - 1
        # A perfect circle has value 0. Higher values indicate irregularity/roughness.
        # This is often called "Form Factor" inverse or "Circularity" inverse related metric.
        # Circularity = 4 * pi * Area / Perimeter^2.
        # Roughness here = (1 / Circularity) - 1.
        
        perimeter = prop.perimeter
        area = prop.area
        
        if area > 0 and perimeter > 0:
            circularity = (4 * math.pi * area) / (perimeter ** 2)
            # Clamp circularity to max 1.0 (perfect circle) to avoid negative roughness due to discrete pixel approximation errors
            if circularity > 1.0:
                circularity = 1.0
            
            # Avoid division by zero if circularity is somehow 0
            if circularity > 0:
                roughness = (1.0 / circularity) - 1.0
                roughness_scores.append(roughness)

    if not roughness_scores:
        return 0.0

    # Return the mean roughness of all nuclei in the image
    return float(np.mean(roughness_scores))
