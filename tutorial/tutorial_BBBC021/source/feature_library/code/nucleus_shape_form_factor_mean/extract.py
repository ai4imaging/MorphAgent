def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    import math

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected 3-channel image, return 0.0
        return 0.0

    # Extract the DAPI channel (Channel 2 - Blue) which corresponds to Nuclei
    # The dataset description specifies: Channel 2 = Blue (B) = DAPI = Nucleus
    nuclei_channel = arr[:, :, 2]

    # Normalize intensity to [0, 1] for processing
    # Although Otsu works on uint8, normalization ensures consistency
    vmax = np.percentile(nuclei_channel, 99.5) if nuclei_channel.size > 0 else 1.0
    if vmax > 0:
        nuclei_channel = nuclei_channel / vmax
    nuclei_channel = np.clip(nuclei_channel, 0.0, 1.0)

    # Determine the labeled mask to use
    labeled_mask = None

    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Assume the first mask is the primary object mask (nuclei or cells)
        # If multiple masks exist, usually the nuclear mask is relevant for nuclear features.
        # However, without explicit metadata on which mask is which, we check the first one.
        mask_input = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == nuclei_channel.shape[:2]:
            # If the mask is already labeled (integer > 1), use it directly
            if np.max(mask_input) > 1:
                labeled_mask = mask_input.astype(int)
            else:
                # If binary, label it
                labeled_mask = label(mask_input > 0)

    # Fallback: Perform on-the-fly segmentation if no mask is provided
    if labeled_mask is None:
        # 1. Thresholding
        try:
            thresh = threshold_otsu(nuclei_channel)
            binary_mask = nuclei_channel > thresh
        except Exception:
            # Fallback for very low contrast/empty images
            binary_mask = nuclei_channel > 0.1

        # 2. Morphological cleanup (remove small noise)
        binary_mask = binary_opening(binary_mask, disk(2))

        # 3. Watershed segmentation to separate touching nuclei
        # Compute distance transform
        distance = ndimage.distance_transform_edt(binary_mask)
        
        # Find peaks in the distance map
        # min_distance=7 corresponds to roughly 7 pixels radius (14px diameter), 
        # reasonable for MCF-7 nuclei at 512x512 resolution
        coords = peak_local_max(distance, min_distance=7, labels=binary_mask)
        
        # Create markers for watershed
        mask_peaks = np.zeros(distance.shape, dtype=bool)
        mask_peaks[tuple(coords.T)] = True
        markers = label(mask_peaks)
        
        # Apply watershed
        labeled_mask = watershed(-distance, markers, mask=binary_mask)

    # Compute Region Properties
    props = regionprops(labeled_mask)

    form_factors = []

    for prop in props:
        # Filter out very small objects (likely noise or debris)
        # Area < 50 pixels is likely too small to be a valid nucleus at this resolution
        if prop.area < 50:
            continue

        # Calculate Form Factor (Circularity)
        # Formula: (4 * pi * Area) / (Perimeter^2)
        # A perfect circle has a value of 1.0.
        # Irregular, lobulated, or elongated shapes have values < 1.0.
        
        area = prop.area
        perimeter = prop.perimeter

        if perimeter == 0:
            continue

        ff = (4 * math.pi * area) / (perimeter ** 2)

        # Due to discrete pixel approximation, FF can slightly exceed 1.0 for small circles.
        # We clip it to 1.0 for physical consistency.
        if ff > 1.0:
            ff = 1.0
            
        form_factors.append(ff)

    # Return the mean form factor
    if len(form_factors) == 0:
        return 0.0
    
    result = np.mean(form_factors)
    return float(result)
