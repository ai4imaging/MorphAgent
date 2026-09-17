def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage import measure, filters, morphology, segmentation
    from scipy import ndimage

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality and select the Nuclear channel (Channel 2 / Blue)
    # Dataset is (512, 512, 3) RGB. Channel 2 is DAPI.
    if arr.ndim == 3 and arr.shape[2] >= 3:
        nuclear_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if passed a single channel image, assume it's the relevant one
        nuclear_channel = arr
    else:
        return 0.0

    # Determine the labeled mask to use
    labeled_mask = None

    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask (assuming it corresponds to nuclei or cells)
        # Ensure it's integer type for labeling
        mask_input = np.asarray(segmentation_masks[0], dtype=np.int32)
        
        # If the mask is already labeled (max > 1), use it directly. 
        # If it's binary (0 and 1), label it.
        if mask_input.max() > 1:
            labeled_mask = mask_input
        else:
            labeled_mask = measure.label(mask_input)
    else:
        # Fallback: Generate segmentation from the nuclear channel
        # 1. Normalize for thresholding
        norm_img = nuclear_channel
        vmax = np.percentile(norm_img, 99) if norm_img.size > 0 else 1.0
        if vmax > 0:
            norm_img = norm_img / vmax
        norm_img = np.clip(norm_img, 0.0, 1.0)

        # 2. Smooth to reduce noise
        smooth_img = ndimage.gaussian_filter(norm_img, sigma=2.0)

        # 3. Threshold (Otsu)
        try:
            thresh = filters.threshold_otsu(smooth_img)
            binary_mask = smooth_img > thresh
        except ValueError:
            # Handle case where image is uniform (e.g., all black)
            return 0.0

        # 4. Clean up (remove small holes and small objects)
        binary_mask = morphology.remove_small_objects(binary_mask, min_size=50)
        binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=50)

        # 5. Label connected components
        labeled_mask = measure.label(binary_mask)

    # Clear objects touching the border
    # Objects cut by the border have artificial straight edges which distort perimeter/area ratio
    labeled_mask = segmentation.clear_border(labeled_mask)

    # Compute region properties
    regions = measure.regionprops(labeled_mask)

    if not regions:
        return 0.0

    compactness_values = []

    for region in regions:
        # Filter out very small artifacts that might have survived or been introduced
        if region.area < 50:
            continue

        # Calculate Compactness: Perimeter^2 / Area
        # A perfect circle has P^2/A = (2*pi*r)^2 / (pi*r^2) = 4*pi^2*r^2 / pi*r^2 = 4*pi ~= 12.57
        # A square has P^2/A = (4s)^2 / s^2 = 16
        # Irregular, blebbed, or fragmented nuclei will have higher values.
        
        # We use region.perimeter which calculates perimeter based on the boundary of the object
        # Note: region.perimeter in skimage is an approximation.
        
        p = region.perimeter
        a = region.area
        
        if a > 0:
            compactness = (p ** 2) / a
            compactness_values.append(compactness)

    if not compactness_values:
        return 0.0

    # Return the mean compactness across all valid nuclei in the image
    result = np.mean(compactness_values)

    return float(result)
