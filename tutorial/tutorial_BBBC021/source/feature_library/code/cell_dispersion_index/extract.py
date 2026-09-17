def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from scipy import spatial
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) RGB TIFF
    if arr.ndim != 3:
        # If not 3D (H, W, C), try to handle or return 0
        if arr.ndim == 2:
            # Assume single channel or projected
            pass
        else:
            return 0.0
    
    # Determine image area for density calculation
    height, width = arr.shape[:2]
    area = height * width

    # Step 1: Obtain Cell Centroids
    # We need coordinates (y, x) for every cell.
    
    centroids = []

    # Check if valid segmentation masks are provided
    # The prompt says masks are passed as *segmentation_masks tuple
    # We prefer the first mask if available (usually nuclei or cell)
    has_mask = False
    if len(segmentation_masks) > 0:
        mask = segmentation_masks[0]
        # Ensure mask is valid (not None and has correct dimensions)
        if mask is not None:
            # Mask might be (H, W) or (H, W, 1)
            if mask.ndim == 3:
                mask = mask.squeeze()
            
            if mask.shape[0] == height and mask.shape[1] == width:
                has_mask = True
                # Get centroids from labeled mask
                # Assuming mask is labeled (0=bg, 1..N=cells)
                # If it's binary, we label it first
                if mask.max() == 1:
                    labeled_mask = label(mask)
                else:
                    labeled_mask = mask.astype(int)
                
                props = regionprops(labeled_mask)
                centroids = [p.centroid for p in props]

    # Fallback: Internal Segmentation on DAPI channel
    if not has_mask or len(centroids) == 0:
        # Channel 2 is DAPI (Blue) in this dataset
        if arr.shape[-1] >= 3:
            dapi_channel = arr[..., 2]
        else:
            # Fallback to mean if channels are weird, or channel 0
            dapi_channel = np.mean(arr, axis=-1)

        # Normalize DAPI for thresholding
        dapi_min = dapi_channel.min()
        dapi_max = dapi_channel.max()
        if dapi_max > dapi_min:
            dapi_norm = (dapi_channel - dapi_min) / (dapi_max - dapi_min)
        else:
            dapi_norm = np.zeros_like(dapi_channel)

        # Simple segmentation pipeline
        # 1. Gaussian blur to smooth noise
        dapi_smooth = ndimage.gaussian_filter(dapi_norm, sigma=2)
        
        # 2. Otsu Thresholding
        try:
            thresh = threshold_otsu(dapi_smooth)
            binary = dapi_smooth > thresh
        except ValueError:
            # Handle case where image is uniform (e.g. all black)
            binary = np.zeros_like(dapi_smooth, dtype=bool)

        # 3. Labeling
        labeled_mask = label(binary)
        
        # 4. Extract centroids
        props = regionprops(labeled_mask)
        centroids = [p.centroid for p in props]

    # Step 2: Calculate Nearest Neighbor Statistics
    num_cells = len(centroids)

    # Edge Case: Not enough cells to define spacing
    if num_cells < 2:
        return 0.0

    # Convert centroids to numpy array
    points = np.array(centroids)

    # Use KDTree for efficient nearest neighbor search
    # We query k=2 because the closest neighbor to a point is itself (distance 0)
    tree = spatial.KDTree(points)
    distances, _ = tree.query(points, k=2)
    
    # The second column (index 1) contains the distance to the nearest *other* point
    nearest_neighbor_distances = distances[:, 1]

    # Calculate Mean Observed Distance (r_A)
    mean_observed_dist = np.mean(nearest_neighbor_distances)

    # Step 3: Calculate Clark-Evans Index (R)
    # R = r_A / r_E
    # r_E (Expected Mean Distance) = 1 / (2 * sqrt(density))
    # density = N / Area
    
    density = num_cells / float(area)
    
    # Avoid division by zero if density is somehow 0 (already handled by num_cells check, but safe)
    if density <= 0:
        return 0.0

    expected_mean_dist = 1.0 / (2.0 * np.sqrt(density))

    # Clark-Evans Index
    # R < 1 implies clustering
    # R = 1 implies random
    # R > 1 implies uniform/dispersed
    if expected_mean_dist == 0:
        return 0.0
        
    clark_evans_index = mean_observed_dist / expected_mean_dist

    return float(clark_evans_index)
