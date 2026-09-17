def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects

    # 1. Data Preparation
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality
    # Expected: (H, W, C) -> (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return 0.0

    # Extract DAPI channel (Channel 2 based on dataset description)
    # Channel 0: Actin (R), Channel 1: Tubulin (G), Channel 2: DAPI (B)
    dapi_channel = arr[:, :, 2]

    # 2. Segmentation Logic
    labeled_mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask (assuming it's a nuclear/cell mask)
        input_mask = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if input_mask.shape == dapi_channel.shape:
            # If mask is already labeled (int), use it. If binary, label it.
            if np.issubdtype(input_mask.dtype, np.integer) and input_mask.max() > 1:
                labeled_mask = input_mask
            else:
                labeled_mask = label(input_mask > 0)
    
    # Fallback: Compute segmentation if no valid mask provided
    if labeled_mask is None:
        # Simple preprocessing
        # Gaussian blur to reduce noise before thresholding
        blurred = ndimage.gaussian_filter(dapi_channel, sigma=2.0)
        
        # Check if image is not completely black
        if np.max(blurred) > 0:
            try:
                thresh = threshold_otsu(blurred)
                binary_mask = blurred > thresh
                # Remove small artifacts (noise)
                binary_mask = remove_small_objects(binary_mask, min_size=50)
                labeled_mask = label(binary_mask)
            except Exception:
                # Fallback if otsu fails (e.g. uniform image)
                return 0.0
        else:
            return 0.0

    # 3. Feature Computation: DAPI Intensity CV
    # CV = std / mean
    
    # Get unique labels (excluding background 0)
    # Using regionprops is convenient for iterating over objects
    regions = regionprops(labeled_mask, intensity_image=dapi_channel)
    
    cv_values = []
    
    for region in regions:
        # Extract pixel intensities for the current nucleus
        # region.image is the binary mask of the object in the bounding box
        # region.intensity_image is the intensity values in the bounding box
        
        # We need the actual pixel values within the mask
        intensities = region.intensity_image[region.image]
        
        if intensities.size > 0:
            mean_val = np.mean(intensities)
            std_val = np.std(intensities)
            
            # Avoid division by zero
            if mean_val > 1e-6:
                cv = std_val / mean_val
                cv_values.append(cv)
            else:
                cv_values.append(0.0)

    # 4. Aggregation
    # Return the mean CV across all nuclei in the image
    if len(cv_values) > 0:
        result = np.mean(cv_values)
    else:
        result = 0.0

    return float(result)
