def extract(img, seg):
    import numpy as np
    from scipy import ndimage

    # Convert image to float for consistent computations
    arr = np.asarray(img, dtype=np.float32)
    
    # Normalize intensity data to [0, 1] (from [0, 9042] based on the dataset description)
    arr /= arr.max()

    # Handle segmentation data
    if not seg:
        return 0.0  # Return zero if no segmentation data is provided
    
    # Retrieve relevant masks
    cell_mask = seg.get("mask_cell")  # Soma region
    bundle_mask = seg.get("mask_bundle")  # Axonal region
    
    if cell_mask is None or bundle_mask is None:
        return 0.0  # If masks are missing, return zero
    
    # Ensure masks are boolean
    cell_mask = cell_mask.astype(bool)
    bundle_mask = bundle_mask.astype(bool)
    
    # Compute mean intensity within the soma region
    soma_intensity_mean = np.mean(arr[cell_mask]) if np.any(cell_mask) else 0.0
    
    # Compute mean intensity within the axonal region
    axon_intensity_mean = np.mean(arr[bundle_mask]) if np.any(bundle_mask) else 0.0
    
    # Avoid division by zero
    if soma_intensity_mean == 0.0:
        return 0.0
    
    # Calculate axon-to-soma Tau intensity ratio
    axon_to_soma_ratio = axon_intensity_mean / soma_intensity_mean
    
    return float(axon_to_soma_ratio)
