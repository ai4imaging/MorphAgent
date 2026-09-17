def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops, perimeter
    from skimage.filters import threshold_otsu
    from skimage.morphology import convex_hull_image
    from scipy import ndimage

    # Convert to appropriate array type if needed, though uint8 is fine for segmentation
    # We need to handle the specific channel for DAPI
    
    # Check input dimensionality
    # Expected: (H, W, C) = (512, 512, 3)
    if img.ndim != 3 or img.shape[2] < 3:
        # Fallback or error handling for unexpected shapes
        return 0.0

    # Extract DAPI channel (Channel 2 / Index 2 based on dataset description)
    dapi_channel = img[:, :, 2]

    # Determine Segmentation Mask
    # Priority: Use provided segmentation mask if available, otherwise compute one
    labeled_mask = None

    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Assuming the first mask corresponds to nuclei or cells
        # If the mask is already labeled (int), use it. If binary, label it.
        mask_input = segmentation_masks[0]
        if mask_input.ndim == 3:
            # If mask is 3D (e.g. same shape as image), take the relevant slice or max projection
            # Often masks are 2D. If 3D, assume channel 0 or max projection.
            if mask_input.shape == img.shape:
                # Try to guess if it's a multi-channel mask or just matches image shape
                # Usually segmentation masks are single channel 2D
                mask_input = mask_input[:, :, 0] # Take first channel if 3D
            else:
                 mask_input = np.max(mask_input, axis=2) # Max project if unknown 3D structure
        
        if np.issubdtype(mask_input.dtype, np.integer):
            # Check if it's binary (0/1 or 0/255) or instance labels
            unique_vals = np.unique(mask_input)
            if len(unique_vals) <= 2:
                labeled_mask = label(mask_input > 0)
            else:
                labeled_mask = mask_input
        else:
            # If float/bool, threshold and label
            labeled_mask = label(mask_input > 0)
    
    # Fallback: Compute segmentation on the fly if no mask provided
    if labeled_mask is None:
        # Preprocessing: Gaussian blur to reduce noise
        blurred = ndimage.gaussian_filter(dapi_channel, sigma=2.0)
        
        # Thresholding (Otsu)
        try:
            thresh = threshold_otsu(blurred)
            binary_mask = blurred > thresh
        except ValueError:
            # Handle case where image is uniform (e.g., all black)
            return 0.0
            
        # Label connected components
        labeled_mask = label(binary_mask)

    # Compute Feature: Boundary Roughness
    # Roughness = Perimeter_real / Perimeter_convex_hull
    
    roughness_values = []
    
    # Iterate over regions
    # We filter out very small regions which are likely noise
    regions = regionprops(labeled_mask)
    
    for region in regions:
        if region.area < 50:
            continue
            
        # 1. Get actual perimeter
        p_real = region.perimeter
        
        # 2. Get convex hull perimeter
        # region.convex_image is the binary convex hull of the region
        # We calculate the perimeter of this binary image
        if region.convex_image.size > 0:
            p_hull = perimeter(region.convex_image)
        else:
            p_hull = 0.0
            
        # Avoid division by zero
        if p_hull > 0:
            # Roughness ratio
            # A perfect circle/ellipse is close to 1.0
            # Blebbing/irregular shapes > 1.0
            r = p_real / p_hull
            roughness_values.append(r)

    # Aggregation
    if not roughness_values:
        return 0.0
        
    # Return the mean roughness across all nuclei in the image
    result = np.mean(roughness_values)

    return float(result)
