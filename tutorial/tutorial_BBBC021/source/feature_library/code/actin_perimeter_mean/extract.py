def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects, binary_closing, disk
    from skimage.segmentation import clear_border, watershed
    from skimage.feature import peak_local_max

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Actin channel (Channel 0)
    # Channel 0 = Red = Actin (Cytoskeleton)
    actin_channel = arr[:, :, 0]

    # Normalize Actin channel to [0, 1]
    vmax = np.percentile(actin_channel, 99.5) if actin_channel.size > 0 else 1.0
    if vmax > 0:
        actin_norm = actin_channel / vmax
    else:
        actin_norm = actin_channel
    actin_norm = np.clip(actin_norm, 0.0, 1.0)

    # Determine the segmentation mask to use
    labeled_mask = None

    # Scenario 1: Segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == actin_norm.shape:
            # If mask is boolean/binary, label it. If integer, assume it's already labeled.
            if mask_input.dtype == bool or np.max(mask_input) == 1:
                labeled_mask = label(mask_input)
            else:
                labeled_mask = mask_input.astype(int)

    # Scenario 2: No segmentation mask provided (Fallback: Compute from Actin channel)
    if labeled_mask is None:
        # 1. Thresholding Actin to get cell body mask
        try:
            thresh_actin = threshold_otsu(actin_norm)
        except Exception:
            thresh_actin = 0.1
        
        binary_mask = actin_norm > thresh_actin
        
        # Clean up binary mask
        binary_mask = binary_closing(binary_mask, disk(2))
        binary_mask = remove_small_objects(binary_mask, min_size=100)
        binary_mask = clear_border(binary_mask)

        # 2. Separate touching cells using Nuclei (Channel 2) as seeds if possible
        nuclei_channel = arr[:, :, 2]
        
        # Normalize nuclei
        n_vmax = np.percentile(nuclei_channel, 99.5) if nuclei_channel.size > 0 else 1.0
        if n_vmax > 0:
            nuclei_norm = nuclei_channel / n_vmax
        else:
            nuclei_norm = nuclei_channel
        
        try:
            thresh_nuc = threshold_otsu(nuclei_norm)
            nuclei_mask = nuclei_norm > thresh_nuc
            nuclei_mask = remove_small_objects(nuclei_mask, min_size=50)
            
            # Generate markers from nuclei
            markers = label(nuclei_mask)
            
            # Watershed segmentation
            # Use negative actin intensity as topographic surface
            labeled_mask = watershed(-actin_norm, markers, mask=binary_mask)
        except Exception:
            # Fallback if watershed fails: just label the binary actin mask
            labeled_mask = label(binary_mask)

    # Compute properties
    if labeled_mask is None or np.max(labeled_mask) == 0:
        return 0.0

    props = regionprops(labeled_mask)
    
    # Extract perimeters
    # Filter out very small artifacts that might have survived
    perimeters = [p.perimeter for p in props if p.area > 50]

    if len(perimeters) == 0:
        return 0.0

    # Calculate mean perimeter
    result = np.mean(perimeters)

    return float(result)
