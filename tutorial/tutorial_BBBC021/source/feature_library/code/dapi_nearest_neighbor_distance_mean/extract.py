def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channel 2 is DAPI (Blue)
    if arr.ndim == 3 and arr.shape[2] == 3:
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if only 2D image provided, assume it's the relevant channel or grayscale
        dapi_channel = arr
    else:
        return 0.0

    # Determine Labeled Mask
    labeled_mask = None
    
    # 1. Try using provided segmentation masks
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == dapi_channel.shape[:2]:
            # If mask is already labeled (int type with values > 1), use it directly
            if np.issubdtype(mask_input.dtype, np.integer) and mask_input.max() > 1:
                labeled_mask = mask_input
            else:
                # If binary, label it
                labeled_mask = label(mask_input > 0)

    # 2. Fallback: Compute segmentation on the fly if no valid mask provided
    if labeled_mask is None:
        # Normalize DAPI for segmentation
        dapi_norm = dapi_channel.copy()
        vmax = np.percentile(dapi_norm, 99.5) if dapi_norm.size > 0 else 1.0
        if vmax > 0:
            dapi_norm = dapi_norm / vmax
        dapi_norm = np.clip(dapi_norm, 0.0, 1.0)
        
        # Smooth to reduce noise
        dapi_smooth = ndimage.gaussian_filter(dapi_norm, sigma=2.0)
        
        try:
            thresh = threshold_otsu(dapi_smooth)
            binary_mask = dapi_smooth > thresh
            # Remove small artifacts
            binary_mask = binary_opening(binary_mask, footprint=disk(2))
            labeled_mask = label(binary_mask)
        except Exception:
            # Fallback if thresholding fails (e.g., empty image)
            return 0.0

    # Extract Centroids
    # regionprops returns a list of properties for each labeled region
    regions = regionprops(labeled_mask)
    
    # Filter out very small objects (noise) that might skew distance metrics
    valid_centroids = []
    min_area = 50  # Minimum pixel area to be considered a nucleus
    
    for props in regions:
        if props.area >= min_area:
            valid_centroids.append(props.centroid)
            
    valid_centroids = np.array(valid_centroids)
    
    # Compute Nearest Neighbor Distances
    num_cells = len(valid_centroids)
    
    # If fewer than 2 cells, distance is undefined/0
    if num_cells < 2:
        return 0.0
        
    # Build KD-Tree for efficient spatial querying
    tree = cKDTree(valid_centroids)
    
    # Query for the 2 nearest neighbors (k=2)
    # The 1st nearest neighbor is the point itself (distance 0)
    # The 2nd nearest neighbor is the actual closest other cell
    distances, indices = tree.query(valid_centroids, k=2)
    
    # Extract the distance to the 2nd neighbor (column index 1)
    # distances shape is (N, 2)
    nearest_neighbor_dists = distances[:, 1]
    
    # Compute Mean
    result = np.mean(nearest_neighbor_dists)

    return float(result)
