def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects
    from scipy import ndimage

    # Convert to appropriate array type if needed, though uint8 is fine for thresholding
    # We ensure we have a numpy array
    img = np.asarray(img)

    # Handle dimensionality and channel selection
    # Dataset is (512, 512, 3) -> (Height, Width, Channels)
    # Channel 0: Actin (Cytoskeleton) - Best for cell shape
    # Channel 1: Tubulin (Microtubules) - Also good for cell shape
    # Channel 2: DAPI (Nucleus) - Not ideal for whole-cell aspect ratio
    
    # Check if image is valid
    if img.ndim < 2:
        return 0.0
    
    # Determine the segmentation mask to use
    labeled_mask = None
    
    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first available mask
        mask_input = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions (H, W)
        if mask_input.shape[:2] == img.shape[:2]:
            # If the mask is already labeled (integer labels > 1), use it directly
            if np.issubdtype(mask_input.dtype, np.integer) and np.max(mask_input) > 1:
                labeled_mask = mask_input
            else:
                # If binary or boolean, label it
                labeled_mask = label(mask_input > 0)
    
    # 2. Fallback: Generate segmentation from image if no valid mask provided
    if labeled_mask is None:
        # Combine Actin (0) and Tubulin (1) to get the best cell body representation
        # If image is (H, W, C)
        if img.ndim == 3 and img.shape[2] >= 2:
            # Use max projection of structural channels
            structure_img = np.maximum(img[..., 0], img[..., 1])
        elif img.ndim == 3:
            # Fallback for unexpected channel count, just use mean
            structure_img = np.mean(img, axis=2)
        else:
            # 2D image
            structure_img = img

        # Smooth to reduce noise
        structure_img = ndimage.gaussian_filter(structure_img, sigma=2)
        
        # Thresholding
        try:
            thresh = threshold_otsu(structure_img)
            binary_mask = structure_img > thresh
        except Exception:
            # Fallback if image is constant
            return 0.0

        # Clean up: remove small artifacts
        binary_mask = remove_small_objects(binary_mask, min_size=50)
        
        # Label connected components
        labeled_mask = label(binary_mask)

    # Calculate properties
    # We need 'major_axis_length' and 'minor_axis_length'
    props = regionprops(labeled_mask)
    
    if not props:
        return 0.0

    aspect_ratios = []
    
    for prop in props:
        major = prop.major_axis_length
        minor = prop.minor_axis_length
        
        # Filter out extremely small objects or artifacts that might have 0 dimensions
        if major == 0:
            continue
            
        # Avoid division by zero. If minor axis is 0 (e.g. 1px wide line), 
        # the aspect ratio is effectively infinite/very large.
        # We use a small epsilon.
        if minor <= 0:
            # Treat as a very elongated object, but cap it to avoid statistical skew
            # or skip if it looks like noise. 
            # Here we skip degenerate objects to keep the mean robust.
            continue
            
        ratio = major / minor
        aspect_ratios.append(ratio)

    if not aspect_ratios:
        return 0.0

    # Calculate mean aspect ratio
    result = np.mean(aspect_ratios)
    
    return float(result)
