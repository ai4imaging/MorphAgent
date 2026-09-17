def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from scipy.spatial import distance
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3)
    if arr.ndim != 3:
        # If not 3D, we can't reliably extract the DAPI channel by index 2
        # If it's 2D (512, 512), treat it as a single channel image
        if arr.ndim == 2:
            dapi_channel = arr
        else:
            return 0.0
    else:
        # Standard format: (H, W, C)
        # Channel 2 is DAPI (Nucleus), which is best for counting cells
        if arr.shape[2] >= 3:
            dapi_channel = arr[:, :, 2]
        else:
            # Fallback if fewer channels
            dapi_channel = np.mean(arr, axis=2)

    # Determine the labeled mask
    labeled_mask = None

    # 1. Try to use provided segmentation masks
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == arr.shape[:2]:
            # If mask is already labeled (int type with multiple values), use it
            if np.issubdtype(mask_input.dtype, np.integer) and mask_input.max() > 1:
                labeled_mask = mask_input
            else:
                # If binary, label it
                labeled_mask = label(mask_input > 0)
    
    # 2. Fallback: Generate segmentation from DAPI channel if no mask provided
    if labeled_mask is None:
        # Normalize DAPI channel
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
        except Exception:
            # Fallback for very low contrast/empty images
            binary_mask = dapi_smooth > 0.1

        # Remove noise
        binary_mask = remove_small_objects(binary_mask, min_size=50)

        # Watershed for separation
        distance_map = ndimage.distance_transform_edt(binary_mask)
        # Find peaks
        coords = peak_local_max(distance_map, min_distance=7, labels=binary_mask)
        mask_peaks = np.zeros(distance_map.shape, dtype=bool)
        mask_peaks[tuple(coords.T)] = True
        markers, _ = ndimage.label(mask_peaks)
        
        labeled_mask = watershed(-distance_map, markers, mask=binary_mask)

    # Extract centroids
    props = regionprops(labeled_mask)
    
    # If no cells found, return 0
    if len(props) == 0:
        return 0.0
    
    # Get centroids as (N, 2) array
    centroids = np.array([p.centroid for p in props])
    
    # If only one cell, it has 0 neighbors
    if len(centroids) < 2:
        return 0.0

    # Compute pairwise Euclidean distances
    # Result is an (N, N) matrix
    dist_matrix = distance.cdist(centroids, centroids, metric='euclidean')

    # Define neighbor radius
    # Heuristic: In a 512x512 image of MCF-7 cells, nuclei are typically 20-40 pixels in diameter.
    # "Touching or within a small radius" implies immediate proximity.
    # 50 pixels is a reasonable radius to capture immediate neighbors (approx 1-2 cell diameters).
    radius = 50.0

    # Count neighbors within radius
    # dist_matrix < radius gives a boolean matrix
    # Sum along rows gives count of neighbors (including self)
    neighbor_counts = np.sum(dist_matrix < radius, axis=1)

    # Subtract 1 because a cell is always within radius 0 of itself
    neighbor_counts = neighbor_counts - 1

    # Calculate mean neighbor count
    mean_neighbor_count = np.mean(neighbor_counts)

    return float(mean_neighbor_count)
