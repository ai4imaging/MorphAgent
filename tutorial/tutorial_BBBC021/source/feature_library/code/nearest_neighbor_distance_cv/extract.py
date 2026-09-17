def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy.spatial import cKDTree
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channel 2 is DAPI (Nucleus)
    # If the input is 2D (H, W), assume it's a single channel or pre-processed.
    # If (H, W, C), extract the nuclear channel.
    
    nuclear_channel = None
    
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Standard format: (512, 512, 3) -> Channel 2 is Blue/DAPI
        nuclear_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback for 2D inputs
        nuclear_channel = arr
    else:
        # Unexpected format
        return 0.0

    # Determine Centroids
    centroids = []

    # Strategy 1: Use Segmentation Masks if available
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask = segmentation_masks[0]
        # Ensure mask is labeled (integer labels)
        if mask.ndim == 3:
             # If mask is 3D (e.g. one-hot or z-stack), take max projection or first channel
             # Assuming standard 2D label mask here based on description
             mask = mask.max(axis=0) if mask.shape[0] < mask.shape[1] else mask[:,:,0]
        
        # If mask is boolean or binary, label it
        if mask.dtype == bool or mask.max() == 1:
            labeled_mask = label(mask)
        else:
            labeled_mask = mask.astype(int)
            
        props = regionprops(labeled_mask)
        centroids = [p.centroid for p in props]

    # Strategy 2: Fallback to Image-based Segmentation
    else:
        # Normalize for thresholding
        if nuclear_channel.max() > nuclear_channel.min():
            norm_img = (nuclear_channel - nuclear_channel.min()) / (nuclear_channel.max() - nuclear_channel.min())
        else:
            norm_img = np.zeros_like(nuclear_channel)
            
        # Simple Otsu thresholding to find nuclei
        try:
            thresh = threshold_otsu(norm_img)
            binary = norm_img > thresh
            labeled_mask = label(binary)
            props = regionprops(labeled_mask)
            centroids = [p.centroid for p in props]
        except Exception:
            # Fallback if thresholding fails (e.g., empty image)
            centroids = []

    # Check if we have enough points to calculate nearest neighbors
    # Need at least 2 points to have a neighbor distance > 0
    if len(centroids) < 2:
        return 0.0

    # Convert centroids to numpy array
    points = np.array(centroids)

    # Compute Nearest Neighbor Distances using KD-Tree
    # k=2 because the 1st nearest neighbor is the point itself (distance 0)
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    
    # The second column contains the distance to the nearest *other* point
    nn_distances = distances[:, 1]

    # Compute Statistics
    mean_dist = np.mean(nn_distances)
    std_dist = np.std(nn_distances)

    # Compute Coefficient of Variation (CV)
    # CV = sigma / mu
    if mean_dist == 0:
        return 0.0
    
    cv = std_dist / mean_dist

    return float(cv)
