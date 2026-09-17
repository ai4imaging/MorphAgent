def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import clear_border
    from skimage.morphology import binary_closing, disk, remove_small_objects
    import math

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) where Channel 0 is Actin
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract Actin channel (Channel 0)
        actin_channel = arr[:, :, 0]
    elif arr.ndim == 2:
        # Fallback if passed a single channel image (unlikely based on spec but safe)
        actin_channel = arr
    else:
        # Unexpected format
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] for consistent thresholding
    vmax = np.percentile(actin_channel, 99.5) if actin_channel.size > 0 else 1.0
    if vmax > 0:
        actin_channel = actin_channel / vmax
    actin_channel = np.clip(actin_channel, 0.0, 1.0)

    # Determine Segmentation Mask
    # We need a mask that defines the cell boundaries.
    # Priority: 
    # 1. Provided segmentation mask (if it looks like a cell mask)
    # 2. Computed mask from Actin channel
    
    labeled_mask = None
    
    # Check provided masks
    if len(segmentation_masks) > 0:
        # Heuristic: Use the first mask provided. 
        # In a real scenario, we might check filenames or metadata, 
        # but here we assume the system passes relevant masks.
        # We need to ensure it's a labeled mask (integers).
        candidate_mask = segmentation_masks[0]
        if candidate_mask.shape == actin_channel.shape:
             labeled_mask = candidate_mask.astype(int)

    # If no valid mask provided, compute one from Actin channel
    if labeled_mask is None:
        # 1. Smooth the image to reduce texture noise (actin fibers) and focus on the hull
        # Sigma=2 is chosen to blur internal details while keeping the edge
        blurred = ndimage.gaussian_filter(actin_channel, sigma=2.0)
        
        # 2. Thresholding
        try:
            thresh = threshold_otsu(blurred)
            binary_mask = blurred > thresh
        except Exception:
            # Fallback for empty/uniform images
            return 0.0
            
        # 3. Morphological cleanup
        # Close gaps in the cytoskeleton
        binary_mask = binary_closing(binary_mask, disk(3))
        # Remove small debris (noise)
        binary_mask = remove_small_objects(binary_mask, min_size=100)
        # Clear border objects (their perimeter is artificial)
        binary_mask = clear_border(binary_mask)
        
        # 4. Labeling
        labeled_mask = label(binary_mask)
    else:
        # If using provided mask, still clear borders to ensure valid perimeter calculations
        # Note: If the provided mask is already labeled, clear_border works on it too
        labeled_mask = clear_border(labeled_mask)

    # Feature Computation
    roughness_values = []
    
    props = regionprops(labeled_mask)
    
    for prop in props:
        # Filter tiny objects that might be artifacts
        if prop.area < 50:
            continue
            
        # Calculate Roughness
        # Formula: Perimeter^2 / (4 * pi * Area)
        # A perfect circle has roughness 1.0. Higher values = more irregular/rough.
        # Note: Discrete pixel perimeter is an approximation.
        
        perimeter = prop.perimeter
        area = prop.area
        
        if area > 0:
            roughness = (perimeter ** 2) / (4 * math.pi * area)
            roughness_values.append(roughness)

    # Aggregation
    if len(roughness_values) == 0:
        return 0.0
    
    result = np.mean(roughness_values)

    return float(result)
