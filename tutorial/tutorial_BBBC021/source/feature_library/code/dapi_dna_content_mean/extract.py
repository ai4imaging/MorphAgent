def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects
    from skimage.segmentation import clear_border, watershed
    from skimage.feature import peak_local_max

    # 1. Input Validation and Channel Extraction
    # Dataset Description: (512, 512, 3), Channel 2 = DAPI (Blue)
    # Check if image is valid
    if img is None or img.ndim < 2:
        return 0.0
    
    # Handle dimensions
    # If 3D (H, W, C), extract DAPI channel (index 2)
    if img.ndim == 3 and img.shape[2] == 3:
        dapi_channel = img[:, :, 2]
    elif img.ndim == 2:
        # Fallback: if only 2D provided, assume it is the relevant channel or a projection
        dapi_channel = img
    else:
        # Unexpected shape
        return 0.0

    # Convert to float for precise summation (prevent uint8 overflow)
    dapi_float = dapi_channel.astype(np.float64)

    # 2. Segmentation Logic
    # Determine if we use provided masks or compute one on the fly
    labeled_mask = None

    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask
        mask_input = segmentation_masks[0]
        
        # Ensure mask shape matches image shape (handle potential squeezing)
        if mask_input.shape == dapi_channel.shape:
            # If mask is already labeled (int > 1), use it directly
            if np.max(mask_input) > 1:
                labeled_mask = mask_input.astype(int)
            else:
                # If binary mask, label it
                labeled_mask = label(mask_input > 0)
    
    # Fallback: On-the-fly segmentation if no valid mask provided
    if labeled_mask is None:
        # Preprocessing: Gaussian blur to reduce noise
        blurred = ndimage.gaussian_filter(dapi_float, sigma=2)
        
        # Thresholding
        try:
            thresh = threshold_otsu(blurred)
            binary_mask = blurred > thresh
        except Exception:
            # Fallback for empty/uniform images
            return 0.0

        # Watershed segmentation to separate touching nuclei
        # This is critical for DNA content analysis to distinguish G2 (4N) from two touching G1 (2N+2N)
        distance = ndimage.distance_transform_edt(binary_mask)
        # Find peaks in distance map (centers of nuclei)
        coords = peak_local_max(distance, min_distance=7, labels=binary_mask)
        mask_peaks = np.zeros(distance.shape, dtype=bool)
        mask_peaks[tuple(coords.T)] = True
        markers = label(mask_peaks)
        
        # Apply watershed
        labeled_mask = watershed(-distance, markers, mask=binary_mask)
        
        # Remove artifacts
        # Remove small objects (debris)
        labeled_mask = remove_small_objects(labeled_mask, min_size=50)
        # Clear border objects (incomplete nuclei have incomplete DNA content)
        labeled_mask = clear_border(labeled_mask)

    # 3. Feature Computation: Mean Integrated Intensity
    # Integrated Intensity = Sum of pixel intensities within the object
    
    # Get properties of labeled regions
    regions = regionprops(labeled_mask, intensity_image=dapi_float)
    
    if not regions:
        return 0.0

    integrated_intensities = []
    for region in regions:
        # region.intensity_image gives the pixel values inside the bounding box
        # region.image gives the binary mask inside the bounding box
        # We sum the intensity only where the mask is True
        
        # Alternatively, simpler calculation: mean_intensity * area
        # This is mathematically equivalent to sum(pixels)
        total_intensity = region.mean_intensity * region.area
        integrated_intensities.append(total_intensity)

    if not integrated_intensities:
        return 0.0

    # Compute the mean of the integrated intensities across the population
    result = np.mean(integrated_intensities)

    return float(result)
