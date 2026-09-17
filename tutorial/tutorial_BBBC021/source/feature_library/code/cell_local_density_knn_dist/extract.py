def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    
    # 1. Data Loading and Preprocessing
    # Convert to float32 for processing
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensionality
    # Expected shape is (H, W, C) = (512, 512, 3) or similar 2D multi-channel
    if arr.ndim != 3:
        # If 2D (H, W), treat as single channel
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        else:
            return 0.0

    # 2. Centroid Extraction
    centroids = []

    # Strategy: Use segmentation masks if available, otherwise fallback to DAPI channel segmentation
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask
        mask = segmentation_masks[0]
        
        # Ensure mask is labeled (instance segmentation)
        # If the mask is binary (max value is 1), we label it to separate connected components
        if mask.max() <= 1:
            labeled_mask = label(mask > 0)
        else:
            labeled_mask = mask.astype(int)
            
        props = regionprops(labeled_mask)
        centroids = [p.centroid for p in props]
        
    else:
        # Fallback: Segment nuclei from Channel 2 (Blue/DAPI)
        # Channel 2 is index 2 in RGB (0=R, 1=G, 2=B)
        if arr.shape[-1] >= 3:
            dapi_channel = arr[..., 2]
        else:
            # If fewer than 3 channels, use the last one or the only one
            dapi_channel = arr[..., -1]
            
        # Normalize channel for segmentation
        if dapi_channel.max() > dapi_channel.min():
            dapi_norm = (dapi_channel - dapi_channel.min()) / (dapi_channel.max() - dapi_channel.min())
        else:
            dapi_norm = np.zeros_like(dapi_channel)

        # Apply smoothing
        dapi_smooth = ndimage.gaussian_filter(dapi_norm, sigma=2.0)
        
        # Thresholding
        try:
            thresh = threshold_otsu(dapi_smooth)
            binary_mask = dapi_smooth > thresh
        except ValueError:
            # Handle case where image is uniform (e.g., all black)
            binary_mask = np.zeros_like(dapi_smooth, dtype=bool)

        # Label connected components
        labeled_mask = label(binary_mask)
        props = regionprops(labeled_mask)
        centroids = [p.centroid for p in props]

    # Convert centroids to numpy array
    points = np.array(centroids)
    num_cells = len(points)

    # 3. KNN Distance Calculation
    # We need at least 2 cells to calculate a distance
    if num_cells < 2:
        return 0.0

    # Target k=3 neighbors
    k = 3
    
    # Adjust k if we have fewer cells than k+1 (since the query returns the point itself)
    # If we have 3 cells, we can find at most 2 neighbors.
    if num_cells <= k:
        k = num_cells - 1
        
    if k < 1:
        return 0.0

    # Build KD-Tree
    tree = cKDTree(points)
    
    # Query for k+1 neighbors (the first neighbor is the point itself with dist=0)
    # distances shape: (num_cells, k+1)
    distances, _ = tree.query(points, k=k+1)
    
    # Remove the first column (distance to self, which is 0.0)
    neighbor_distances = distances[:, 1:]
    
    # 4. Feature Computation
    # Compute the mean distance to the k nearest neighbors for each cell
    # This gives a local density metric per cell
    mean_dist_per_cell = np.mean(neighbor_distances, axis=1)
    
    # The final feature is the average of these local metrics across the whole image
    result = np.mean(mean_dist_per_cell)

    return float(result)
