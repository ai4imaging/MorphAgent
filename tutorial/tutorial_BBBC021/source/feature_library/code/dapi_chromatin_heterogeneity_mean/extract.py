def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import shannon_entropy, label, regionprops
    from skimage.filters import threshold_otsu
    from scipy import ndimage

    # 1. Input Validation and Preparation
    # Ensure input is a numpy array
    arr = np.asarray(img)
    
    # Check dimensionality and extract DAPI channel
    # Dataset is (512, 512, 3), Channel 2 is DAPI (Blue)
    if arr.ndim == 3 and arr.shape[2] == 3:
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if single channel passed
        dapi_channel = arr
    else:
        # Unexpected format
        return 0.0

    # 2. Segmentation Logic
    # We need a labeled mask identifying individual nuclei
    labeled_mask = None

    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask
        mask_input = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == dapi_channel.shape[:2]:
            # If the mask is already labeled (max > 1), use it directly
            # If it's binary (max == 1), label it
            if mask_input.max() > 1:
                labeled_mask = mask_input.astype(int)
            else:
                labeled_mask = label(mask_input > 0)

    # Fallback: Compute segmentation on the fly if no valid mask provided
    if labeled_mask is None:
        # Simple preprocessing to reduce noise
        blurred = ndimage.gaussian_filter(dapi_channel.astype(float), sigma=2.0)
        
        # Otsu thresholding
        try:
            thresh = threshold_otsu(blurred)
            binary_mask = blurred > thresh
            
            # Remove small artifacts
            binary_mask = ndimage.binary_opening(binary_mask, structure=np.ones((3,3)))
            
            # Label connected components
            labeled_mask = label(binary_mask)
        except Exception:
            # If thresholding fails (e.g., empty image), return 0.0
            return 0.0

    # 3. Feature Computation: Chromatin Heterogeneity (Entropy)
    # We calculate Shannon entropy for each nucleus and take the mean
    
    heterogeneity_scores = []
    
    # Iterate through each segmented nucleus
    # We pass the original uint8 DAPI channel as the intensity image
    regions = regionprops(labeled_mask, intensity_image=dapi_channel)
    
    for region in regions:
        # Filter out very small regions (debris/noise)
        if region.area < 50:
            continue
            
        # Extract the pixel intensities for this specific nucleus
        # region.image is the binary mask of the object within its bounding box
        # region.intensity_image is the intensity within the bounding box
        # We select only the pixels belonging to the mask
        nucleus_pixels = region.intensity_image[region.image]
        
        if nucleus_pixels.size == 0:
            continue

        # Calculate Shannon Entropy
        # Entropy measures the "randomness" or texture complexity of the intensity distribution
        # High entropy -> High heterogeneity (condensed/fragmented chromatin)
        # Low entropy -> Low heterogeneity (diffuse/smooth chromatin)
        # We use base 2 for bits
        score = shannon_entropy(nucleus_pixels, base=2)
        heterogeneity_scores.append(score)

    # 4. Aggregation
    if not heterogeneity_scores:
        return 0.0
        
    result = np.mean(heterogeneity_scores)

    return float(result)
