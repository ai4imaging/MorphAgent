def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu, gaussian
    from skimage.morphology import binary_closing, binary_opening, disk
    from skimage.segmentation import clear_border

    # Convert to appropriate array type
    # Image is (512, 512, 3), uint8
    arr = np.asarray(img)
    
    # Check dimensionality
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected 3-channel image, return 0.0
        return 0.0

    # Extract the Nucleus Channel (Channel 2 - Blue - DAPI)
    # Based on dataset description: Ch0=Actin, Ch1=Tubulin, Ch2=DAPI
    nucleus_channel = arr[:, :, 2]

    # Determine the labeled mask
    labeled_mask = None

    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Assume the first mask corresponds to the nuclei or cells
        # Check if it matches spatial dimensions
        mask_candidate = segmentation_masks[0]
        if mask_candidate.shape[:2] == nucleus_channel.shape[:2]:
            # If it's a labeled mask (int type with values > 1), use it directly
            # If it's a binary mask, label it
            if np.issubdtype(mask_candidate.dtype, np.integer) and np.max(mask_candidate) > 1:
                labeled_mask = mask_candidate
            else:
                labeled_mask = label(mask_candidate > 0)

    # 2. Fallback: Perform on-the-fly segmentation if no valid mask provided
    if labeled_mask is None:
        # Normalize channel for segmentation
        norm_channel = nucleus_channel.astype(np.float32) / 255.0
        
        # Apply Gaussian blur to reduce noise
        blurred = gaussian(norm_channel, sigma=2.0)
        
        # Determine threshold
        try:
            thresh = threshold_otsu(blurred)
        except ValueError:
            # Handle case where image is uniform (e.g., all black)
            return 0.0
            
        # Create binary mask
        binary_mask = blurred > thresh
        
        # Morphological cleanup
        # Close holes inside nuclei
        binary_mask = binary_closing(binary_mask, disk(2))
        # Remove small noise
        binary_mask = binary_opening(binary_mask, disk(1))
        
        # Clear nuclei touching the border (their perimeter/area would be incorrect)
        binary_mask = clear_border(binary_mask)
        
        # Label the objects
        labeled_mask = label(binary_mask)

    # Compute properties
    props = regionprops(labeled_mask)
    
    compactness_values = []
    
    for prop in props:
        # Filter out very small artifacts
        if prop.area < 50:
            continue
            
        # Calculate Compactness: Perimeter^2 / Area
        # A perfect circle has compactness ~ 12.57 (4 * pi)
        # Irregular shapes have higher values
        if prop.area > 0:
            comp = (prop.perimeter ** 2) / prop.area
            compactness_values.append(comp)

    # Return the mean compactness
    if len(compactness_values) == 0:
        return 0.0
    
    result = np.mean(compactness_values)
    
    return float(result)
