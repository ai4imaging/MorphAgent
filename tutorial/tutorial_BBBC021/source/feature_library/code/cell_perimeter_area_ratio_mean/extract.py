def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu, gaussian
    from skimage.morphology import remove_small_objects, binary_closing, disk
    from skimage.segmentation import clear_border

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) where Channel 0 = Actin (Red), 1 = Tubulin, 2 = DAPI
    # If the image is 2D (H, W), we assume it's a single channel projection or grayscale.
    # If it's 3D (H, W, C), we extract the Actin channel (Channel 0).
    
    actin_channel = None
    
    if arr.ndim == 2:
        # If only 2D, assume it's the relevant data or a projection
        actin_channel = arr
    elif arr.ndim == 3:
        if arr.shape[2] == 3: # (H, W, C) standard format
            actin_channel = arr[:, :, 0] # Channel 0 is Actin
        elif arr.shape[0] == 3: # (C, H, W) channel-first format
            actin_channel = arr[0, :, :]
        else:
            # Fallback: take mean projection if channels are ambiguous
            actin_channel = np.mean(arr, axis=2)
    else:
        return 0.0

    # Normalize Actin channel for segmentation
    # Range check and normalization to [0, 1]
    if actin_channel.max() > actin_channel.min():
        actin_channel = (actin_channel - actin_channel.min()) / (actin_channel.max() - actin_channel.min())
    else:
        # Flat image, no features
        return 0.0

    # Determine Segmentation Mask
    # We prioritize provided masks. If none, we segment the Actin channel.
    labeled_mask = None

    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask. 
        # We assume the first mask corresponds to cells or cytoplasm if available.
        # If multiple masks are provided, typically index 0 is cells/cytoplasm and index 1 is nuclei in many pipelines.
        # We check dimensions to ensure compatibility.
        mask_candidate = segmentation_masks[0]
        if mask_candidate.shape == actin_channel.shape:
            labeled_mask = mask_candidate.astype(int)
        elif mask_candidate.ndim == 2 and actin_channel.ndim == 2:
             # Shape mismatch but both 2D - likely resizing needed or invalid mask. 
             # We skip invalid masks and fall back to self-segmentation.
             pass
    
    # Fallback: Self-Segmentation on Actin Channel
    if labeled_mask is None:
        # 1. Smooth to reduce noise
        blurred = gaussian(actin_channel, sigma=2)
        
        # 2. Threshold (Otsu)
        try:
            thresh = threshold_otsu(blurred)
            binary_mask = blurred > thresh
        except ValueError:
            # Image might be uniform
            return 0.0
            
        # 3. Morphological cleanup
        # Close gaps in the cytoskeleton
        binary_mask = binary_closing(binary_mask, disk(2))
        # Remove small artifacts (noise)
        binary_mask = remove_small_objects(binary_mask, min_size=100)
        # Clear cells touching the border (their perimeter/area ratio is invalid because they are cut off)
        binary_mask = clear_border(binary_mask)
        
        # 4. Label
        labeled_mask = label(binary_mask)

    # Compute Feature
    # Feature: Mean of (Perimeter^2 / Area)
    
    regions = regionprops(labeled_mask)
    
    if not regions:
        return 0.0

    ratios = []
    for region in regions:
        area = region.area
        perimeter = region.perimeter
        
        # Filter out tiny regions that might be artifacts or have unstable ratios
        if area > 50 and perimeter > 0:
            # Calculate P^2 / A
            # A circle has P^2/A = (2*pi*r)^2 / (pi*r^2) = 4*pi^2*r^2 / pi*r^2 = 4*pi ~= 12.57
            # Complex shapes have higher values.
            ratio = (perimeter ** 2) / area
            ratios.append(ratio)

    if not ratios:
        return 0.0

    result = np.mean(ratios)
    
    return float(result)
