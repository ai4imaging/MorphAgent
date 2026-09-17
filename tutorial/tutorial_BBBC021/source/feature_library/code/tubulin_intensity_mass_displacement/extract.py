def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) where Channel 1 is Tubulin
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract Tubulin channel (Index 1)
        tubulin_channel = arr[:, :, 1]
    elif arr.ndim == 2:
        # If 2D, assume it's already a single channel (fallback)
        tubulin_channel = arr
    else:
        return 0.0

    # Intensity normalization
    # Normalize to [0, 1] for stability, though center_of_mass works with raw values
    # We clip to avoid negative values if any preprocessing occurred
    tubulin_channel = np.clip(tubulin_channel, 0, None)
    
    # Determine the mask to use
    labeled_mask = None
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask
        mask_input = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == tubulin_channel.shape[:2]:
            # If mask is already labeled (int > 1), use it directly
            if np.issubdtype(mask_input.dtype, np.integer) and mask_input.max() > 1:
                labeled_mask = mask_input
            else:
                # If binary mask, label it
                labeled_mask = label(mask_input > 0)
    
    # Fallback: If no valid mask provided, generate one from the image
    if labeled_mask is None:
        # Simple background subtraction and thresholding
        try:
            # Use Tubulin channel for segmentation fallback
            thresh = threshold_otsu(tubulin_channel)
            binary_mask = tubulin_channel > thresh
            labeled_mask = label(binary_mask)
        except Exception:
            # If thresholding fails (e.g. constant image), return 0
            return 0.0

    # Get unique labels (excluding background 0)
    unique_labels = np.unique(labeled_mask)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) == 0:
        return 0.0

    # Compute Geometric Centroids
    # center_of_mass with input=None or input=ones calculates the geometric center of the mask region
    # We create a binary representation for geometric calculation
    binary_region = (labeled_mask > 0).astype(np.float32)
    # Actually, ndimage.center_of_mass calculates geometric center if 'input' is the mask itself (binary 0/1)
    # But strictly, we pass the labels. 
    # To get geometric center: input should be an array of ones where the mask is.
    # However, standard practice:
    # Geometric center: Center of mass of the binary mask.
    # Weighted center: Center of mass of the intensity image.
    
    # Efficient calculation for all labels at once
    # 1. Geometric Centroids (unweighted)
    # We pass an array of ones as 'input' to treat mass as uniform
    ones_img = np.ones_like(tubulin_channel)
    geo_centers = ndimage.center_of_mass(input=ones_img, labels=labeled_mask, index=unique_labels)
    
    # 2. Intensity-Weighted Centroids
    # We pass the actual tubulin intensity as 'input'
    weighted_centers = ndimage.center_of_mass(input=tubulin_channel, labels=labeled_mask, index=unique_labels)

    # Calculate displacements
    displacements = []
    
    # Ensure outputs are lists of tuples even if single object
    if not isinstance(geo_centers, list):
        geo_centers = [geo_centers]
    if not isinstance(weighted_centers, list):
        weighted_centers = [weighted_centers]

    for gc, wc in zip(geo_centers, weighted_centers):
        # Handle potential NaNs if a label exists but has 0 sum intensity (unlikely but possible)
        if np.any(np.isnan(gc)) or np.any(np.isnan(wc)):
            continue
            
        # Euclidean distance
        dist = np.sqrt((gc[0] - wc[0])**2 + (gc[1] - wc[1])**2)
        displacements.append(dist)

    if not displacements:
        return 0.0

    # Return the mean displacement across all cells in the image
    result = np.mean(displacements)

    return float(result)
