def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu, gaussian
    from skimage.morphology import remove_small_objects, binary_closing, disk
    from skimage.segmentation import clear_border

    # 1. Input Validation and Channel Selection
    # The dataset is (512, 512, 3). Channel 2 is DAPI (Nuclei).
    if img.ndim == 3 and img.shape[2] >= 3:
        # Extract the nuclear channel (Blue/Channel 2)
        nuclear_channel = img[:, :, 2]
    elif img.ndim == 2:
        # Fallback if a single channel image is passed (unlikely based on spec but safe)
        nuclear_channel = img
    else:
        return 0.0

    # Convert to float for processing if needed, though skimage handles types well.
    # Gaussian smoothing helps reduce noise before thresholding.
    # Sigma=1.0 is generally good for nuclei in 512x512 images.
    nuclear_smooth = gaussian(nuclear_channel, sigma=1.0)

    # 2. Segmentation Logic
    labeled_nuclei = None

    # Check if a valid segmentation mask is provided in *segmentation_masks
    # The prompt implies masks might be passed. If so, we look for one matching the image shape.
    if len(segmentation_masks) > 0:
        # Iterate to find a suitable mask. 
        # We assume the first mask matching the spatial dimensions is the nuclear mask 
        # or a general cell mask we can use.
        for mask in segmentation_masks:
            if mask is not None and mask.shape[:2] == nuclear_channel.shape[:2]:
                # Ensure it's labeled (integers)
                if mask.dtype == bool:
                    labeled_nuclei = label(mask)
                else:
                    labeled_nuclei = mask.astype(int)
                break
    
    # If no external mask provided, perform self-segmentation
    if labeled_nuclei is None:
        try:
            # Otsu's thresholding
            thresh = threshold_otsu(nuclear_smooth)
            binary_mask = nuclear_smooth > thresh
            
            # Morphological cleanup
            # Fill small holes
            binary_mask = binary_closing(binary_mask, disk(2))
            # Remove small artifacts (noise)
            binary_mask = remove_small_objects(binary_mask, min_size=50)
            # Clear nuclei touching the border (their shapes are incomplete and skew roundness)
            binary_mask = clear_border(binary_mask)
            
            # Label the regions
            labeled_nuclei = label(binary_mask)
        except Exception:
            # Fallback if thresholding fails (e.g., empty image)
            return 0.0

    # 3. Feature Computation: Mean Roundness
    # Roundness = (4 * pi * Area) / (Perimeter^2)
    
    regions = regionprops(labeled_nuclei)
    
    if not regions:
        return 0.0

    roundness_values = []
    
    for region in regions:
        area = region.area
        perimeter = region.perimeter
        
        # Filter out extremely small objects or single pixels to avoid division by zero
        # or meaningless shapes
        if perimeter > 0 and area > 5:
            # Calculate roundness
            # A perfect circle has roundness 1.0
            # Irregular shapes have roundness < 1.0
            r = (4 * np.pi * area) / (perimeter ** 2)
            roundness_values.append(r)
            
    # 4. Aggregation
    if not roundness_values:
        return 0.0
        
    result = np.mean(roundness_values)
    
    return float(result)
