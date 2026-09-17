def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), uint8
    # Channel 2 is DAPI (Nuclei), which is critical for cell centroid detection
    
    # Check if image is valid
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not 3-channel, try to handle if it's 2D (perhaps a single channel passed incorrectly)
        # But strictly based on dataset description, it should be (H, W, 3)
        if arr.ndim == 2:
            # Assume single channel input is DAPI-like for fallback
            dapi_channel = arr
            image_area = arr.shape[0] * arr.shape[1]
        else:
            return 0.0
    else:
        # Extract DAPI channel (Channel 2 - Blue)
        dapi_channel = arr[:, :, 2]
        image_area = arr.shape[0] * arr.shape[1]

    # --- Step 1: Obtain Labeled Nuclei ---
    labeled_mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first mask provided (assumed to be nuclei or cell mask)
        mask_input = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == dapi_channel.shape:
            # If mask is already labeled (int), use it. If boolean/binary, label it.
            if np.issubdtype(mask_input.dtype, np.integer) and mask_input.max() > 1:
                labeled_mask = mask_input
            else:
                labeled_mask = label(mask_input > 0)
    
    # Fallback: Generate mask from DAPI channel if no valid segmentation provided
    if labeled_mask is None:
        # Normalize DAPI channel for thresholding
        dapi_norm = dapi_channel
        vmax = np.percentile(dapi_norm, 99.5) if dapi_norm.size > 0 else 1.0
        if vmax > 0:
            dapi_norm = dapi_norm / vmax
        dapi_norm = np.clip(dapi_norm, 0.0, 1.0)
        
        # Simple thresholding pipeline
        try:
            thresh = threshold_otsu(dapi_norm)
            binary_mask = dapi_norm > thresh
            
            # Remove small noise (optional but good for stability)
            binary_mask = ndimage.binary_opening(binary_mask, structure=np.ones((3,3)))
            
            labeled_mask = label(binary_mask)
        except Exception:
            # Fallback if otsu fails (e.g., empty image)
            return 0.0

    # --- Step 2: Extract Centroids ---
    # regionprops returns a list of properties for each labeled region
    regions = regionprops(labeled_mask)
    
    # Get centroids: list of (row, col) tuples
    centroids = [r.centroid for r in regions]
    centroids = np.array(centroids)
    
    N = len(centroids)

    # --- Step 3: Compute Clark-Evans Index ---
    
    # Edge cases: Need at least 2 points to calculate nearest neighbor distance
    if N < 2:
        # If 0 or 1 cell, clustering is undefined. 
        # Returning 1.0 (random) or 0.0 is a design choice. 
        # 0.0 might imply "clustered" incorrectly, but NaN is often problematic for downstream ML.
        # We return 1.0 to indicate "neutral/undefined" spatial pattern.
        return 1.0

    # Build KD-Tree for efficient nearest neighbor search
    tree = cKDTree(centroids)
    
    # Query for 2 nearest neighbors (k=2). 
    # The first neighbor is the point itself (distance=0), the second is the actual nearest neighbor.
    # distances is (N, 2), indices is (N, 2)
    distances, _ = tree.query(centroids, k=2)
    
    # Extract the distance to the nearest neighbor (column index 1)
    nearest_neighbor_distances = distances[:, 1]
    
    # Mean Observed Distance (r_obs)
    mean_observed_dist = np.mean(nearest_neighbor_distances)
    
    # Density (rho)
    # rho = Number of points / Area
    rho = N / float(image_area)
    
    if rho <= 0:
        return 1.0
        
    # Expected Mean Distance for Random Distribution (r_exp)
    # Formula for 2D Poisson process: 1 / (2 * sqrt(rho))
    mean_expected_dist = 1.0 / (2.0 * np.sqrt(rho))
    
    # Clark-Evans Index R = r_obs / r_exp
    if mean_expected_dist == 0:
        return 0.0
        
    R = mean_observed_dist / mean_expected_dist
    
    # Interpretation:
    # R < 1: Clustered (aggregated)
    # R = 1: Random
    # R > 1: Regular (dispersed/uniform)

    return float(R)
