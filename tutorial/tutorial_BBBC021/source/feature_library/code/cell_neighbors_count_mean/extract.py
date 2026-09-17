def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from scipy import spatial
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk

    # --- 1. Data Loading and Preprocessing ---
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3:
        # If 2D (512, 512), treat as single channel or projection
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)
        else:
            return 0.0
    
    # --- 2. Segmentation Strategy ---
    # We need a labeled mask to identify individual cells/nuclei.
    # Priority: Use provided segmentation masks -> Fallback to DAPI channel segmentation.
    
    labeled_mask = None
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first available mask
        # Ensure it's integer labeled
        mask_input = np.asarray(segmentation_masks[0])
        if mask_input.ndim == 2:
            if np.issubdtype(mask_input.dtype, np.integer):
                labeled_mask = mask_input
            else:
                # If boolean or float mask, label it
                labeled_mask = label(mask_input > 0)
        elif mask_input.ndim == 3:
             # If 3D mask, take max projection or slice (assuming 2D analysis)
             mask_2d = np.max(mask_input, axis=0) if mask_input.shape[0] < mask_input.shape[1] else np.max(mask_input, axis=-1)
             labeled_mask = label(mask_2d > 0)

    # Fallback: Generate segmentation from DAPI (Channel 2, index 2) if no mask provided
    if labeled_mask is None:
        # Check if channel 2 exists
        if arr.shape[-1] >= 3:
            dapi_channel = arr[..., 2]
        else:
            dapi_channel = np.mean(arr, axis=-1) # Fallback to mean intensity
            
        # Normalize DAPI for segmentation
        dapi_min, dapi_max = dapi_channel.min(), dapi_channel.max()
        if dapi_max > dapi_min:
            dapi_norm = (dapi_channel - dapi_min) / (dapi_max - dapi_min)
        else:
            dapi_norm = np.zeros_like(dapi_channel)

        # Simple segmentation pipeline
        try:
            thresh = threshold_otsu(dapi_norm)
            binary = dapi_norm > thresh
            # Clean up noise and separate touching objects slightly
            binary = binary_opening(binary, footprint=disk(2))
            labeled_mask = label(binary)
        except Exception:
            return 0.0

    # --- 3. Feature Computation: Neighbor Counting ---
    
    # Get properties of labeled regions
    props = regionprops(labeled_mask)
    
    # Filter out very small artifacts (noise)
    valid_props = [p for p in props if p.area > 10]
    num_cells = len(valid_props)
    
    if num_cells < 2:
        return 0.0
    
    # Extract centroids
    centroids = np.array([p.centroid for p in valid_props])
    
    # Use Delaunay Triangulation to find natural neighbors
    # This is robust for both touching and non-touching (nuclei) objects
    try:
        tri = spatial.Delaunay(centroids)
    except Exception:
        # Fallback for collinear points or other triangulation failures
        return 0.0

    # Build adjacency list (set of neighbors for each cell index)
    # The Delaunay triangulation consists of simplices (triangles). 
    # Vertices in the same simplex are neighbors.
    neighbors = {i: set() for i in range(num_cells)}
    
    # Iterate over all simplices (triangles)
    # simplex is an array of 3 indices [p1, p2, p3]
    for simplex in tri.simplices:
        for i in range(3):
            p1_idx = simplex[i]
            p2_idx = simplex[(i + 1) % 3]
            p3_idx = simplex[(i + 2) % 3]
            
            # Add neighbors (undirected graph)
            neighbors[p1_idx].add(p2_idx)
            neighbors[p1_idx].add(p3_idx)
            neighbors[p2_idx].add(p1_idx)
            neighbors[p2_idx].add(p3_idx)
            neighbors[p3_idx].add(p1_idx)
            neighbors[p3_idx].add(p2_idx)

    # Filter neighbors by distance
    # Delaunay connects everything, even distant cells across gaps.
    # We apply a distance threshold to define "local" neighbors.
    # Heuristic: 80 pixels is reasonable for 512x512 images of cells (approx 2-3 cell diameters)
    MAX_DISTANCE = 80.0 
    
    neighbor_counts = []
    
    for i in range(num_cells):
        current_centroid = centroids[i]
        valid_neighbors_count = 0
        
        for neighbor_idx in neighbors[i]:
            neighbor_centroid = centroids[neighbor_idx]
            dist = np.linalg.norm(current_centroid - neighbor_centroid)
            
            if dist <= MAX_DISTANCE:
                valid_neighbors_count += 1
        
        neighbor_counts.append(valid_neighbors_count)

    # --- 4. Aggregation ---
    if not neighbor_counts:
        return 0.0
        
    result = np.mean(neighbor_counts)
    
    return float(result)
