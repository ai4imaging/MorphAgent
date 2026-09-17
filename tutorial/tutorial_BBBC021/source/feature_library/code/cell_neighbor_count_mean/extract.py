def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from scipy import spatial
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu

    # 1. Data Loading and Preprocessing
    # Ensure image is float32 for calculations
    img_arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality (512, 512, 3) expected
    if img_arr.ndim != 3:
        return 0.0
    
    # 2. Define Cell Centroids
    # We need to identify individual cells (nuclei) to determine their locations.
    # Strategy: Use provided segmentation mask if available; otherwise, segment the DAPI channel.
    
    labeled_mask = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask = segmentation_masks[0]
        # Ensure mask is 2D
        if mask.ndim == 3:
            mask = np.max(mask, axis=2) # Project if 3D
        elif mask.ndim == 2:
            pass
        else:
            mask = None # Invalid shape
            
        if mask is not None:
            # If mask is already labeled (int), use it. If binary, label it.
            if np.issubdtype(mask.dtype, np.integer) and mask.max() > 1:
                labeled_mask = mask
            else:
                labeled_mask = label(mask > 0)

    # Fallback: Segment DAPI channel (Channel 2) if no mask provided
    if labeled_mask is None:
        # Extract DAPI channel (Index 2)
        if img_arr.shape[-1] > 2:
            dapi = img_arr[..., 2]
        else:
            # Fallback for unexpected channel count, use mean
            dapi = np.mean(img_arr, axis=-1)
            
        # Normalize DAPI for segmentation
        dapi_norm = (dapi - np.min(dapi)) / (np.max(dapi) - np.min(dapi) + 1e-6)
        
        # Smooth to reduce noise
        dapi_smooth = ndimage.gaussian_filter(dapi_norm, sigma=2)
        
        # Threshold
        try:
            thresh = threshold_otsu(dapi_smooth)
            binary = dapi_smooth > thresh
        except:
            return 0.0 # Empty image or uniform intensity
            
        # Label
        labeled_mask = label(binary)

    # 3. Extract Centroids
    # Filter small objects (noise)
    props = regionprops(labeled_mask)
    centroids = []
    min_area = 20 # Minimum pixel area to be considered a cell
    
    for prop in props:
        if prop.area >= min_area:
            centroids.append(prop.centroid) # Returns (row, col)
            
    centroids = np.array(centroids)
    num_cells = len(centroids)

    # 4. Compute Neighbors
    
    # Edge Case: No cells or single cell
    if num_cells < 2:
        return 0.0
    
    neighbor_counts = np.zeros(num_cells)
    
    # Distance threshold for "biological" neighbor (pixels)
    # 512x512 image. A typical nucleus might be ~20-40 pixels wide.
    # Neighbors should be within ~2-3 cell diameters.
    # Let's set a conservative threshold to prune long Delaunay edges across empty space.
    distance_threshold = 80.0 

    # Strategy A: Delaunay Triangulation (Efficient for N > 4)
    if num_cells >= 4:
        try:
            tri = spatial.Delaunay(centroids)
            
            # The triangulation contains simplices (triangles).
            # We need to extract unique edges and check their lengths.
            # Indptr/indices structure of Delaunay is complex, simpler to iterate simplices.
            
            # Adjacency list: set of neighbor indices for each cell index
            adjacency = {i: set() for i in range(num_cells)}
            
            for simplex in tri.simplices:
                # Simplex is [idx1, idx2, idx3]
                # Check edges: (0,1), (1,2), (2,0)
                edges = [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]
                
                for idx_a, idx_b in edges:
                    # Calculate distance
                    dist = np.linalg.norm(centroids[idx_a] - centroids[idx_b])
                    
                    if dist <= distance_threshold:
                        adjacency[idx_a].add(idx_b)
                        adjacency[idx_b].add(idx_a)
            
            # Count neighbors
            for i in range(num_cells):
                neighbor_counts[i] = len(adjacency[i])
                
        except Exception:
            # Fallback to distance matrix if Delaunay fails (e.g., collinear points)
            dists = spatial.distance.pdist(centroids)
            sq_dists = spatial.distance.squareform(dists)
            # Count how many are within threshold (excluding self, distance 0)
            # logical_and(>0, <=threshold)
            mask = (sq_dists > 0) & (sq_dists <= distance_threshold)
            neighbor_counts = np.sum(mask, axis=1)

    # Strategy B: Brute Force Distance Matrix (for small N)
    else:
        dists = spatial.distance.pdist(centroids)
        sq_dists = spatial.distance.squareform(dists)
        mask = (sq_dists > 0) & (sq_dists <= distance_threshold)
        neighbor_counts = np.sum(mask, axis=1)

    # 5. Compute Mean
    mean_neighbors = np.mean(neighbor_counts)
    
    return float(mean_neighbors)
