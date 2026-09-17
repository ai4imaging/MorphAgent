def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from scipy.spatial import cKDTree
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import opening, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) - 2D composite image
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green)
    # Channel 2: DAPI (Blue) - Nuclei
    
    # We need the nuclei channel for cell packing density
    # If the image is (H, W, C), we extract channel 2.
    if arr.ndim == 3 and arr.shape[2] == 3:
        nuclei_channel = arr[..., 2]
    elif arr.ndim == 2:
        # Fallback if only one channel provided, assume it's relevant
        nuclei_channel = arr
    else:
        return 0.0

    # Determine the labeled mask to use
    labeled_mask = None

    # Check if segmentation masks are available
    # Based on description, masks might be passed. We prioritize the nuclei mask.
    # Usually, if multiple masks are present, one corresponds to nuclei.
    # Without specific metadata on which mask is which in the *args, we inspect them.
    # Often mask 0 is nuclei or mask 1 is cells. We look for the one with more objects 
    # (nuclei are usually distinct, cells might touch) or simply use the first one available.
    
    if len(segmentation_masks) > 0:
        # Try to find a valid mask
        for mask in segmentation_masks:
            if mask is not None:
                # Ensure mask is 2D
                if mask.ndim == 3:
                    # If mask is 3D (e.g. one-hot or RGB), take max projection or first channel
                    m = mask[..., 0] if mask.shape[-1] < 5 else np.max(mask, axis=0)
                else:
                    m = mask
                
                # Check if it's a labeled mask (int) or binary
                m = m.astype(int)
                if m.max() > 1:
                    labeled_mask = m
                    break # Found a labeled mask
                elif m.max() == 1:
                    # It's binary, label it
                    labeled_mask = label(m)
                    break
    
    # Fallback: Perform on-the-fly segmentation on DAPI channel if no mask provided
    if labeled_mask is None:
        # Normalize DAPI channel
        dapi = nuclei_channel
        vmax = np.percentile(dapi, 99.5) if dapi.size > 0 else 1.0
        if vmax > 0:
            dapi = dapi / vmax
        dapi = np.clip(dapi, 0.0, 1.0)
        
        # Simple segmentation pipeline
        # 1. Gaussian blur to reduce noise
        dapi_smooth = ndimage.gaussian_filter(dapi, sigma=2)
        
        # 2. Thresholding
        try:
            thresh = threshold_otsu(dapi_smooth)
            binary = dapi_smooth > thresh
        except Exception:
            return 0.0 # Failed to threshold (e.g. empty image)
            
        # 3. Morphological opening to separate touching blobs slightly
        binary = opening(binary, disk(2))
        
        # 4. Labeling
        labeled_mask = label(binary)

    # Extract Centroids
    # regionprops returns a list of properties for each labeled region
    props = regionprops(labeled_mask)
    
    # We need at least 2 cells to calculate a neighbor distance
    if len(props) < 2:
        return 0.0
    
    # Get centroids (y, x)
    centroids = np.array([p.centroid for p in props])
    
    # Calculate Nearest Neighbor Distances
    # Use KDTree for efficient query
    tree = cKDTree(centroids)
    
    # Query for the 2 nearest neighbors
    # k=2 because the 1st nearest neighbor is the point itself (distance 0)
    # The 2nd nearest neighbor is the actual closest other cell
    distances, indices = tree.query(centroids, k=2)
    
    # The result `distances` is shape (N, 2). 
    # Column 0 is distance to self (0.0), Column 1 is distance to nearest neighbor.
    nearest_neighbor_distances = distances[:, 1]
    
    # Compute the feature: Mean Distance to Nearest Neighbor
    # Feature description: "Estimates the local density of cells by calculating the mean distance to the nearest neighbor"
    mean_nn_distance = np.mean(nearest_neighbor_distances)
    
    return float(mean_nn_distance)
