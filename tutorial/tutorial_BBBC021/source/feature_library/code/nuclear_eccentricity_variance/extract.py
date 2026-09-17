def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, binary_closing, disk
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    # 1. Data Preparation & Channel Extraction
    # Convert to appropriate array type if needed, though uint8 is fine for most ops
    # We need Channel 2 (Blue) for Nuclei (DAPI)
    # Image shape is (512, 512, 3)
    if img.ndim == 3 and img.shape[2] >= 3:
        nuclear_channel = img[:, :, 2]
    elif img.ndim == 2:
        # Fallback if single channel passed (unlikely based on spec but safe)
        nuclear_channel = img
    else:
        return 0.0

    # 2. Segmentation Logic
    labeled_nuclei = None

    # Scenario A: Segmentation Masks Provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Assume the first mask corresponds to the nuclei or is a general cell mask
        # If multiple masks exist, we prioritize the one that likely matches nuclei
        # However, usually the order is consistent. We'll take the first one.
        mask_input = segmentation_masks[0]
        
        # Ensure it's a labeled mask (integers for instances)
        if mask_input.ndim == 2:
            if np.max(mask_input) > 1:
                # Already instance labeled
                labeled_nuclei = mask_input.astype(int)
            else:
                # Binary mask, need to label connected components
                labeled_nuclei = label(mask_input > 0)
        else:
            # Fallback for unexpected mask dimensions
            labeled_nuclei = None

    # Scenario B: No Mask Provided (On-the-fly Segmentation)
    if labeled_nuclei is None:
        # Preprocessing: Gaussian blur to reduce noise
        # Normalize to 0-1 for processing
        nuc_float = nuclear_channel.astype(np.float32) / 255.0
        blurred = ndimage.gaussian_filter(nuc_float, sigma=2)

        # Thresholding
        try:
            thresh = threshold_otsu(blurred)
            binary_mask = blurred > thresh
        except ValueError:
            # Handle case where image is uniform (e.g. all black)
            return 0.0

        # Morphological cleanup
        # Open to remove small noise, Close to fill holes
        binary_mask = binary_opening(binary_mask, disk(2))
        binary_mask = binary_closing(binary_mask, disk(2))

        # Watershed separation for touching nuclei
        # Calculate distance transform
        distance = ndimage.distance_transform_edt(binary_mask)
        
        # Find peaks in distance map (centers of nuclei)
        # min_distance ensures we don't over-segment
        coords = peak_local_max(distance, min_distance=7, labels=binary_mask)
        mask_peaks = np.zeros(distance.shape, dtype=bool)
        mask_peaks[tuple(coords.T)] = True
        markers, _ = ndimage.label(mask_peaks)

        # Apply watershed
        labeled_nuclei = watershed(-distance, markers, mask=binary_mask)

    # 3. Feature Calculation (Per Object)
    # If no objects found, return 0.0
    if labeled_nuclei.max() == 0:
        return 0.0

    # Extract properties
    props = regionprops(labeled_nuclei)
    
    eccentricities = []
    for prop in props:
        # Filter small artifacts (e.g., < 20 pixels area)
        if prop.area < 20:
            continue
        
        # Eccentricity is a property in regionprops
        # 0 = circle, 1 = line
        eccentricities.append(prop.eccentricity)

    # 4. Aggregation
    if not eccentricities:
        return 0.0

    # Calculate Variance
    ecc_variance = np.var(eccentricities)

    return float(ecc_variance)
