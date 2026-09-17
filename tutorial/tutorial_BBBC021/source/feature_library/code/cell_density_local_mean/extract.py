def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from scipy import spatial
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    
    # Define parameters
    k_neighbors = 3  # Number of neighbors to consider for local density
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality and channel selection
    # Dataset is (512, 512, 3) RGB. Channel 2 (Blue) is DAPI (Nuclei).
    # We need nuclei centroids for density calculation.
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract DAPI channel for fallback segmentation
        dapi_channel = arr[..., 2]
    elif arr.ndim == 2:
        # If passed as single channel, assume it's relevant
        dapi_channel = arr
    else:
        # Unexpected format
        return 0.0

    # Determine Centroids
    centroids = []

    # Strategy 1: Use provided segmentation masks
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask = segmentation_masks[0]
        # Ensure mask is labeled (not just binary)
        if mask.max() == 1 and mask.ndim == 2:
            labeled_mask = label(mask)
        else:
            labeled_mask = mask.astype(int)
            
        props = regionprops(labeled_mask)
        centroids = [p.centroid for p in props]

    # Strategy 2: Fallback to on-the-fly segmentation if no mask provided or mask is empty
    if not centroids:
        # Normalize DAPI channel for segmentation
        if dapi_channel.max() > 0:
            norm_dapi = dapi_channel / dapi_channel.max()
        else:
            norm_dapi = dapi_channel
            
        # Simple segmentation pipeline
        # 1. Gaussian blur to reduce noise
        blurred = ndimage.gaussian_filter(norm_dapi, sigma=2)
        
        # 2. Thresholding
        try:
            thresh = threshold_otsu(blurred)
            binary = blurred > thresh
        except ValueError:
            # Handle case where image is uniform (e.g., all black)
            return 0.0
            
        # 3. Labeling
        labeled_mask = label(binary)
        props = regionprops(labeled_mask)
        centroids = [p.centroid for p in props]

    # Convert centroids to numpy array
    points = np.array(centroids)
    num_cells = len(points)

    # Edge Case: Not enough cells to compute density
    if num_cells < 2:
        return 0.0

    # Adjust k if we have fewer cells than the target k_neighbors
    # We need at least k neighbors, so total cells must be at least k + 1
    current_k = min(k_neighbors, num_cells - 1)
    
    if current_k < 1:
        return 0.0

    # Build KD-Tree for efficient nearest neighbor search
    tree = spatial.cKDTree(points)

    # Query for k+1 nearest neighbors
    # k=current_k+1 because the first neighbor is the point itself (distance 0)
    distances, _ = tree.query(points, k=current_k + 1)

    # distances is shape (num_cells, current_k + 1)
    # The first column (index 0) is the distance to self (approx 0.0), discard it
    neighbor_distances = distances[:, 1:]

    # Calculate the mean distance to the k nearest neighbors for each cell
    # This gives a local density metric per cell (in pixels)
    local_means = np.mean(neighbor_distances, axis=1)

    # Aggregate to global feature: Mean of all local means
    # This represents the average "crowding" distance across the image
    global_mean_distance = np.mean(local_means)

    return float(global_mean_distance)
