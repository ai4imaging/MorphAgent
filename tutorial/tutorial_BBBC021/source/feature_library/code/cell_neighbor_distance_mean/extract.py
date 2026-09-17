def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from scipy import spatial
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset description: (Height, Width, Channels) = (512, 512, 3)
    # Channel 2 is DAPI (Nuclei), which is best for defining cell centers.
    
    # Check for valid input shape
    if arr.ndim != 3:
        return 0.0
    
    # Extract the DAPI channel (Channel 2) for fallback segmentation
    # Assuming (H, W, C) order based on description
    if arr.shape[-1] == 3:
        dapi_channel = arr[..., 2]
    elif arr.shape[0] == 3: # Handle potential (C, H, W) case just in case
        dapi_channel = arr[2, ...]
    else:
        # Fallback: use mean projection if channel structure is ambiguous
        dapi_channel = np.mean(arr, axis=-1)

    # Determine the labeled mask to use
    labeled_mask = None

    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_candidate = segmentation_masks[0]
        # Ensure mask is 2D
        if mask_candidate.ndim == 3:
            mask_candidate = np.max(mask_candidate, axis=-1) # Project if 3D
        
        # If mask is binary, label it. If integer, assume it's already labeled.
        if mask_candidate.dtype == bool or (np.unique(mask_candidate).size <= 2 and np.max(mask_candidate) == 1):
            labeled_mask = label(mask_candidate)
        else:
            labeled_mask = mask_candidate.astype(int)

    # 2. Fallback: Compute segmentation on DAPI channel if no mask provided
    if labeled_mask is None:
        # Normalize DAPI channel for segmentation
        if dapi_channel.max() > dapi_channel.min():
            dapi_norm = (dapi_channel - dapi_channel.min()) / (dapi_channel.max() - dapi_channel.min())
        else:
            dapi_norm = np.zeros_like(dapi_channel)
        
        # Smooth
        dapi_smooth = ndimage.gaussian_filter(dapi_norm, sigma=2)
        
        # Threshold
        try:
            thresh = threshold_otsu(dapi_smooth)
            binary_mask = dapi_smooth > thresh
        except ValueError: # Handle empty images
            binary_mask = np.zeros_like(dapi_smooth, dtype=bool)
            
        # Clean up
        binary_mask = binary_opening(binary_mask, disk(2))
        
        # Label
        labeled_mask = label(binary_mask)

    # Extract centroids
    regions = regionprops(labeled_mask)
    centroids = [r.centroid for r in regions]
    
    # Convert to numpy array
    points = np.array(centroids)
    num_cells = len(points)

    # Calculate neighbor distances
    # We need at least 2 cells to calculate a distance
    if num_cells < 2:
        return 0.0
    
    # Define k for k-nearest neighbors (default k=3)
    # If fewer cells than k+1, adjust k to be N-1 (all other cells)
    k = 3
    if num_cells <= k:
        k = num_cells - 1
        
    # Use KDTree for efficient nearest neighbor search
    tree = spatial.cKDTree(points)
    
    # Query for k+1 neighbors (the first one is the point itself with dist=0)
    # distances shape: (N, k+1)
    distances, _ = tree.query(points, k=k+1)
    
    # Exclude the first column (distance to self, which is 0.0)
    neighbor_distances = distances[:, 1:]
    
    # Calculate the mean distance to the k nearest neighbors for each cell
    mean_dist_per_cell = np.mean(neighbor_distances, axis=1)
    
    # Calculate the global mean across all cells
    global_mean_dist = np.mean(mean_dist_per_cell)

    return float(global_mean_dist)
