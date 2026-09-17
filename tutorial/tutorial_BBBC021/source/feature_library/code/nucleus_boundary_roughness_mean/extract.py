def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage import measure, morphology, filters, segmentation
    from scipy import ndimage

    # 1. Input Validation and Preparation
    # Ensure image is at least 2D. If 3D, check if it's (H, W, C) or (Z, H, W) based on description.
    # Dataset description says (512, 512, 3).
    img = np.asarray(img)
    
    if img.ndim < 2:
        return 0.0
    
    # Extract the Nucleus Channel (Channel 2 / Blue based on dataset description)
    # If the image is (H, W, C) with C=3
    if img.ndim == 3 and img.shape[2] == 3:
        nucleus_channel = img[..., 2]
    # If the image is (C, H, W) or (Z, H, W) - though unlikely given description, handle gracefully
    elif img.ndim == 3 and img.shape[0] == 3:
        nucleus_channel = img[0, ...] # Fallback or specific channel logic
    elif img.ndim == 2:
        nucleus_channel = img
    else:
        # Unexpected shape, try to use the last channel or just the array itself
        nucleus_channel = img[..., -1] if img.shape[-1] <= 3 else img

    # 2. Determine Segmentation Mask
    labeled_mask = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches spatial dimensions of the nucleus channel
        if mask_input.shape[-2:] == nucleus_channel.shape[-2:]:
            # If it's already labeled (int), use it. If boolean/binary, label it.
            if np.issubdtype(mask_input.dtype, np.integer) and mask_input.max() > 1:
                labeled_mask = mask_input
            else:
                labeled_mask = measure.label(mask_input > 0)
        elif mask_input.ndim == nucleus_channel.ndim and mask_input.shape == nucleus_channel.shape:
             if np.issubdtype(mask_input.dtype, np.integer) and mask_input.max() > 1:
                labeled_mask = mask_input
             else:
                labeled_mask = measure.label(mask_input > 0)

    # 3. Fallback Segmentation (if no mask provided)
    if labeled_mask is None:
        # Normalize intensity for thresholding
        nuc_float = nucleus_channel.astype(np.float32)
        if nuc_float.max() > 0:
            nuc_float /= nuc_float.max()
        
        # Smooth to reduce noise for cleaner boundaries
        nuc_smooth = ndimage.gaussian_filter(nuc_float, sigma=2.0)
        
        # Thresholding (Otsu)
        try:
            thresh = filters.threshold_otsu(nuc_smooth)
            binary_mask = nuc_smooth > thresh
        except ValueError: # Handle empty images
            return 0.0
            
        # Morphological cleanup
        # Remove small artifacts (noise)
        binary_mask = morphology.remove_small_objects(binary_mask, min_size=50)
        # Fill holes inside nuclei
        binary_mask = ndimage.binary_fill_holes(binary_mask)
        
        # Label the objects
        labeled_mask = measure.label(binary_mask)

    # 4. Feature Computation: Nucleus Boundary Roughness
    # We calculate the Isoperimetric Quotient (Inverse) or Shape Factor
    # Roughness = Perimeter^2 / (4 * pi * Area)
    # Value of 1.0 is a perfect circle. Higher values indicate irregularity/roughness.
    
    # Clear objects touching the border to avoid artificial straight edges
    labeled_mask = segmentation.clear_border(labeled_mask)
    
    props = measure.regionprops(labeled_mask)
    
    roughness_values = []
    
    for prop in props:
        area = prop.area
        perimeter = prop.perimeter
        
        # Filter out very small objects where perimeter calculation is discrete and unstable
        if area < 50:
            continue
            
        if area > 0:
            # Calculate roughness
            # Formula: P^2 / (4 * pi * A)
            # A perfect circle has P = 2*pi*r, A = pi*r^2
            # (2*pi*r)^2 / (4*pi*pi*r^2) = 4*pi^2*r^2 / 4*pi^2*r^2 = 1
            roughness = (perimeter ** 2) / (4 * np.pi * area)
            roughness_values.append(roughness)

    # 5. Aggregation
    if not roughness_values:
        return 0.0
        
    result = np.mean(roughness_values)
    
    return float(result)
