def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import clear_border, watershed
    from skimage.feature import peak_local_max
    from skimage.morphology import remove_small_objects

    # 1. Data Preparation
    # Convert to appropriate array type if needed, though uint8 is fine for basic indexing
    # Extract the Nuclear Channel (Channel 2 - Blue - DAPI)
    # Shape check: (512, 512, 3)
    if img.ndim == 3 and img.shape[2] >= 3:
        nuclear_channel = img[:, :, 2]
    elif img.ndim == 2:
        # Fallback if single channel passed
        nuclear_channel = img
    else:
        return 0.0

    # 2. Segmentation Logic
    labeled_mask = None

    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure it matches image spatial dimensions
        if mask_input.shape[:2] == nuclear_channel.shape[:2]:
            # If the mask is already labeled (max > 1), use it directly
            # If it's binary (max == 1), label it
            if mask_input.max() > 1:
                labeled_mask = mask_input.astype(int)
            elif mask_input.max() == 1:
                labeled_mask = label(mask_input)
    
    # Fallback: Perform on-the-fly segmentation if no valid mask provided
    if labeled_mask is None:
        # Preprocessing: Gaussian blur to reduce noise
        # Normalize to float for filtering
        img_float = nuclear_channel.astype(np.float32)
        blurred = ndimage.gaussian_filter(img_float, sigma=1.0)
        
        # Thresholding (Otsu)
        try:
            thresh_val = threshold_otsu(blurred)
            binary_mask = blurred > thresh_val
        except ValueError:
            # Handle case where image is uniform (e.g., all black)
            return 0.0

        # Morphological cleanup
        # Fill holes
        binary_mask = ndimage.binary_fill_holes(binary_mask)
        # Remove small artifacts (noise) - e.g., objects smaller than 50 pixels
        binary_mask = remove_small_objects(binary_mask, min_size=50)

        # Instance Separation (Watershed)
        # This is critical for clustered nuclei in MCF-7 cells
        distance = ndimage.distance_transform_edt(binary_mask)
        
        # Find peaks in distance map (centers of nuclei)
        # min_distance ensures we don't over-segment
        coords = peak_local_max(distance, min_distance=7, labels=binary_mask)
        mask_peaks = np.zeros(distance.shape, dtype=bool)
        mask_peaks[tuple(coords.T)] = True
        markers = label(mask_peaks)
        
        # Apply watershed
        labeled_mask = watershed(-distance, markers, mask=binary_mask)

    # 3. Post-Processing: Clear Border
    # Nuclei touching the border are often cut, resulting in artificial shapes 
    # (e.g., a circle becomes a semi-circle with high eccentricity).
    # We remove them to ensure unbiased shape analysis.
    labeled_mask = clear_border(labeled_mask)

    # 4. Feature Calculation: Eccentricity
    # Extract properties
    regions = regionprops(labeled_mask)
    
    if not regions:
        return 0.0

    eccentricities = []
    for region in regions:
        # Filter out very small regions that might be debris surviving the process
        if region.area > 50:
            eccentricities.append(region.eccentricity)

    # 5. Aggregation
    if not eccentricities:
        return 0.0
        
    result = np.mean(eccentricities)

    return float(result)
