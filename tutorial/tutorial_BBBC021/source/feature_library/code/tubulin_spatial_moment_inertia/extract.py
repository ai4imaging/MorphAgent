def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops

    # 1. Data Loading and Validation
    # Ensure image is float for calculations
    img_arr = np.asarray(img, dtype=np.float32)

    # Check dimensions: Expecting (H, W, C) = (512, 512, 3)
    # If 2D (H, W), return 0.0 as we need specific channels
    if img_arr.ndim != 3 or img_arr.shape[2] < 2:
        return 0.0

    # 2. Channel Selection
    # Channel 1 is Tubulin (Green) based on dataset description
    tubulin_img = img_arr[:, :, 1]

    # Normalize intensity to [0, 1] for stability
    # Use robust max to avoid hot pixels skewing normalization
    max_val = np.percentile(tubulin_img, 99.9)
    if max_val > 0:
        tubulin_img = tubulin_img / max_val
    tubulin_img = np.clip(tubulin_img, 0, 1)

    # 3. Mask Handling
    # We need to identify individual cells to calculate the moment of inertia relative to *each cell's* centroid.
    # Calculating it over the whole image would be dominated by the spatial layout of cells, not their morphology.
    
    labeled_mask = None
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use provided segmentation mask
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == tubulin_img.shape:
            # If mask is already labeled (int), use it. If binary, label it.
            if np.issubdtype(mask_input.dtype, np.integer) and mask_input.max() > 1:
                labeled_mask = mask_input
            else:
                labeled_mask = label(mask_input > 0)
    
    # Fallback: Generate mask if none provided
    if labeled_mask is None:
        # Simple Otsu-like thresholding or percentile based
        # Tubulin background is usually dark.
        threshold = np.percentile(tubulin_img, 90) # Conservative threshold to get main cell bodies
        binary_mask = tubulin_img > threshold
        # Clean up noise
        binary_mask = ndimage.binary_opening(binary_mask, structure=np.ones((3,3)))
        labeled_mask = label(binary_mask)

    # 4. Feature Calculation: Moment of Inertia (Radius of Gyration)
    # Formula: R_g = sqrt( sum( intensity_i * distance_i^2 ) / sum( intensity_i ) )
    # This quantifies the spatial spread of intensity relative to the object's centroid.
    
    props = regionprops(labeled_mask, intensity_image=tubulin_img)
    
    moments_of_inertia = []

    for prop in props:
        # Skip very small artifacts
        if prop.area < 50:
            continue

        # Get the intensity patch and coordinates for this cell
        # regionprops provides a slice of the bounding box
        intensity_patch = prop.image_intensity
        
        # Local centroid within the bounding box
        # weighted_local_centroid returns (row, col) relative to the slice
        try:
            yc, xc = prop.weighted_local_centroid
        except (ValueError, AttributeError):
            # Fallback to geometric centroid if intensity is uniform/zero
            yc, xc = prop.local_centroid

        # Create coordinate grids for the patch
        # shape is (rows, cols)
        rows, cols = intensity_patch.shape
        y_indices, x_indices = np.indices((rows, cols))

        # Calculate squared distances from centroid
        # r^2 = (x - xc)^2 + (y - yc)^2
        squared_distances = (x_indices - xc)**2 + (y_indices - yc)**2

        # Calculate weighted sum of squared distances (Second Moment)
        # Sum( I * r^2 )
        weighted_sum_sq_dist = np.sum(intensity_patch * squared_distances)
        
        # Total intensity mass
        total_intensity = np.sum(intensity_patch)

        if total_intensity > 0:
            # Radius of Gyration (normalized spatial dispersion)
            # This is the standard "Moment of Inertia" metric in image analysis for spread
            # It represents the RMS distance of the intensity from the center.
            rg = np.sqrt(weighted_sum_sq_dist / total_intensity)
            moments_of_inertia.append(rg)

    # 5. Aggregation
    # Return the mean value across all cells in the image
    if not moments_of_inertia:
        return 0.0
        
    result = np.mean(moments_of_inertia)

    return float(result)
