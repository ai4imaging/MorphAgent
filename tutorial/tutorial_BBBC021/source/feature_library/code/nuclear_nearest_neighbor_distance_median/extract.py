def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from scipy import spatial
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Channel 2 is DAPI (Nucleus)
        nuclear_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if single channel passed
        nuclear_channel = arr
    else:
        return 0.0

    # Intensity normalization for segmentation logic
    # Normalize to [0, 1] for processing
    vmax = np.percentile(nuclear_channel, 99.5) if nuclear_channel.size > 0 else 1.0
    if vmax > 0:
        norm_channel = nuclear_channel / vmax
    else:
        norm_channel = nuclear_channel
    norm_channel = np.clip(norm_channel, 0.0, 1.0)

    # Determine Labeled Mask
    labeled_mask = None

    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == nuclear_channel.shape[:2]:
            # If mask is already labeled (max > 1), use it directly
            if np.max(mask_input) > 1:
                labeled_mask = mask_input.astype(int)
            # If mask is binary, label it
            elif np.max(mask_input) == 1:
                labeled_mask = label(mask_input)
    
    # Fallback: Compute segmentation if no valid mask provided
    if labeled_mask is None:
        # 1. Thresholding
        try:
            thresh = threshold_otsu(norm_channel)
            binary = norm_channel > thresh
        except Exception:
            # Fallback for very low contrast/empty images
            binary = norm_channel > 0.1

        # 2. Watershed to split touching nuclei (crucial for accurate centroids)
        distance = ndimage.distance_transform_edt(binary)
        # Find peaks in distance map (centers of nuclei)
        # min_distance=7 roughly corresponds to radius of MCF-7 nuclei in pixels at this resolution
        coords = peak_local_max(distance, min_distance=7, labels=binary)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers = label(mask)
        
        # Apply watershed
        labeled_mask = watershed(-distance, markers, mask=binary)

    # Extract Centroids
    # regionprops returns a list of properties. We need 'centroid'.
    props = regionprops(labeled_mask)
    
    # If no nuclei found, return 0.0
    if len(props) < 2:
        return 0.0

    # Collect centroids into a (N, 2) array
    centroids = np.array([p.centroid for p in props])

    # Calculate Nearest Neighbor Distances
    # Use KDTree for efficient spatial query
    tree = spatial.cKDTree(centroids)
    
    # Query for the 2 nearest neighbors (k=2)
    # The 1st neighbor is the point itself (distance 0), the 2nd is the actual nearest neighbor
    distances, indices = tree.query(centroids, k=2)
    
    # The second column (index 1) contains the distance to the nearest neighbor
    nearest_neighbor_distances = distances[:, 1]

    # Calculate Median
    result = np.median(nearest_neighbor_distances)

    return float(result)
