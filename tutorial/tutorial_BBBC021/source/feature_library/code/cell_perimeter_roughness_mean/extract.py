def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops, perimeter
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, disk, convex_hull_image
    from scipy import ndimage

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channels: 0=Actin, 1=Tubulin, 2=DAPI
    # If the image is 2D (H, W), treat it as single channel.
    # If 3D (H, W, C), we need to select the appropriate channel for cell boundary analysis.
    
    # Channel Selection for Morphology:
    # Channel 0 (Actin) is best for cell boundaries/shape.
    # Channel 1 (Tubulin) is also good for general cell body.
    # Channel 2 (DAPI) is nuclei, not suitable for cell perimeter roughness.
    
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Use Actin (ch0) primarily, potentially combined with Tubulin (ch1) for robustness
        # Max projection of Actin and Tubulin captures the full cytoskeleton extent
        image_for_segmentation = np.maximum(arr[..., 0], arr[..., 1])
    elif arr.ndim == 2:
        image_for_segmentation = arr
    else:
        return 0.0

    # Intensity normalization
    vmax = np.percentile(image_for_segmentation, 99.5) if image_for_segmentation.size > 0 else 1.0
    if vmax > 0:
        image_for_segmentation = image_for_segmentation / vmax
    image_for_segmentation = np.clip(image_for_segmentation, 0.0, 1.0)

    # Handle segmentation masks
    labeled_mask = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == arr.shape[:2]:
            if mask_input.ndim == 3:
                # If mask is 3D (e.g. one-hot or stacked), take max or first channel
                mask_input = mask_input[..., 0]
            
            # If mask is boolean or binary, label it. If integer, assume instance segmentation.
            if mask_input.dtype == bool or np.max(mask_input) == 1:
                labeled_mask = label(mask_input)
            else:
                labeled_mask = mask_input.astype(int)

    # Fallback: Generate segmentation if no mask provided
    if labeled_mask is None:
        # Smooth to reduce noise
        smoothed = ndimage.gaussian_filter(image_for_segmentation, sigma=2)
        
        # Threshold
        try:
            thresh = threshold_otsu(smoothed)
            binary = smoothed > thresh
        except Exception:
            binary = smoothed > 0.1 # Fallback threshold
            
        # Morphological cleanup
        binary = binary_closing(binary, disk(3))
        
        # Label
        labeled_mask = label(binary)

    # Compute Feature: Cell Perimeter Roughness Mean
    # Formula: Perimeter_actual / Perimeter_convex_hull
    
    regions = regionprops(labeled_mask)
    roughness_values = []
    
    for region in regions:
        # Filter small artifacts
        if region.area < 100:
            continue
            
        # 1. Actual Perimeter
        actual_perimeter = region.perimeter
        if actual_perimeter == 0:
            continue
            
        # 2. Convex Hull Perimeter
        # region.convex_image is the binary convex hull of the region
        # We calculate the perimeter of this binary mask
        hull_image = region.convex_image
        
        # Calculate perimeter of the convex hull
        # We use skimage.measure.perimeter on the binary hull image
        hull_perimeter = perimeter(hull_image)
        
        if hull_perimeter <= 0:
            continue
            
        # 3. Calculate Ratio
        # A perfect convex shape has ratio ~1.0
        # Rough shapes (blebbing, protrusions) have ratio > 1.0
        roughness = actual_perimeter / hull_perimeter
        roughness_values.append(roughness)

    # Aggregate
    if not roughness_values:
        return 0.0
        
    result = np.mean(roughness_values)
    
    return float(result)
