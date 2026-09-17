def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import regionprops, label
    from skimage.filters import threshold_otsu

    # 1. Data Loading and Preprocessing
    # Ensure image is float for calculations
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality and extract Tubulin channel (Channel 1)
    # Expected shape: (512, 512, 3)
    if arr.ndim == 3 and arr.shape[2] == 3:
        tubulin_channel = arr[..., 1]  # Channel 1 is Tubulin (Green)
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on spec, but safe)
        tubulin_channel = arr
    else:
        return 0.0

    # Normalize intensity to [0, 1] for stability
    # Although Moment of Inertia is ratio-based, normalization helps with thresholding fallback
    vmax = np.percentile(tubulin_channel, 99.5) if tubulin_channel.size > 0 else 1.0
    if vmax > 0:
        tubulin_channel = tubulin_channel / vmax
    tubulin_channel = np.clip(tubulin_channel, 0.0, 1.0)

    # 2. Segmentation Handling
    labeled_mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first mask provided. In BBBC021, masks are typically cell masks or nuclei masks.
        # For Tubulin dispersion, a cell mask is ideal. We assume the first mask is the most relevant.
        mask_input = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == tubulin_channel.shape[:2]:
            # If mask is not integer labeled, label it
            if not np.issubdtype(mask_input.dtype, np.integer):
                labeled_mask = label(mask_input > 0)
            else:
                labeled_mask = mask_input
    
    # Fallback: If no valid mask provided, generate one from the Tubulin channel
    if labeled_mask is None:
        try:
            thresh = threshold_otsu(tubulin_channel)
            binary_mask = tubulin_channel > thresh
            # Clean up noise
            binary_mask = ndimage.binary_opening(binary_mask, structure=np.ones((3,3)))
            labeled_mask = label(binary_mask)
        except Exception:
            return 0.0

    # 3. Feature Calculation: Moment of Inertia Mean
    # Formula: Sum(Intensity * Distance^2) / Sum(Intensity)
    # Distance is Euclidean distance from pixel to the geometric centroid of the cell
    
    props = regionprops(labeled_mask, intensity_image=tubulin_channel)
    
    moments = []
    
    for prop in props:
        # Get intensity values within the bounding box
        # regionprops intensity_image gives the intensity values inside the bbox
        # prop.image gives the binary mask inside the bbox
        
        # Extract the intensity patch and mask patch for the single cell
        intensity_patch = prop.intensity_image
        mask_patch = prop.image
        
        if intensity_patch.size == 0 or np.sum(intensity_patch) == 0:
            continue
            
        # Get local coordinates of pixels within the object
        # coords returns (row, col) coordinates relative to the full image, 
        # but calculating relative to bbox is more efficient for distance
        
        # Get indices of pixels that are part of the cell
        y_indices, x_indices = np.nonzero(mask_patch)
        intensities = intensity_patch[y_indices, x_indices]
        
        # Calculate local centroid within the patch
        # prop.centroid is global (row, col). prop.bbox is (min_row, min_col, max_row, max_col)
        # Local centroid = Global Centroid - Bbox Top-Left
        local_centroid_y = prop.centroid[0] - prop.bbox[0]
        local_centroid_x = prop.centroid[1] - prop.bbox[1]
        
        # Calculate squared Euclidean distances from each pixel to the centroid
        # d^2 = (y - cy)^2 + (x - cx)^2
        dist_sq = (y_indices - local_centroid_y)**2 + (x_indices - local_centroid_x)**2
        
        # Calculate Moment of Inertia
        # Weighted sum of squared distances
        weighted_dist_sq_sum = np.sum(intensities * dist_sq)
        total_intensity = np.sum(intensities)
        
        if total_intensity > 0:
            # This is effectively the squared radius of gyration weighted by intensity
            moi = weighted_dist_sq_sum / total_intensity
            moments.append(moi)

    # 4. Aggregation
    if not moments:
        return 0.0
        
    result = np.mean(moments)
    
    return float(result)
