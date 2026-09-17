def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import label

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality: Expecting (H, W, C) = (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Tubulin channel (Channel 1 - Green)
    tubulin_channel = arr[:, :, 1]

    # Normalize intensity
    vmax = np.percentile(tubulin_channel, 99.5) if tubulin_channel.size > 0 else 1.0
    if vmax > 0:
        tubulin_channel = tubulin_channel / vmax
    tubulin_channel = np.clip(tubulin_channel, 0.0, 1.0)

    # Determine segmentation mask
    # If segmentation masks are provided, use the first one (assuming it's a cell/nuclei mask)
    # If not, generate a simple mask using Otsu thresholding on the tubulin channel itself
    labels = None
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask = segmentation_masks[0]
        # Ensure mask is 2D
        if mask.ndim == 3:
            mask = mask.max(axis=2) # Project if 3D
        if mask.shape != tubulin_channel.shape:
             # Fallback if shapes don't match (unlikely given specs but good for safety)
             val = threshold_otsu(tubulin_channel)
             mask = tubulin_channel > val
             labels = label(mask)
        else:
            # If the mask is already labeled (int), use it directly. If boolean, label it.
            if np.issubdtype(mask.dtype, np.integer):
                labels = mask
            else:
                labels = label(mask > 0)
    else:
        # Fallback: Create mask from tubulin channel
        try:
            val = threshold_otsu(tubulin_channel)
            mask = tubulin_channel > val
            labels = label(mask)
        except Exception:
            return 0.0

    # Calculate spatial moment variance for each cell
    props = regionprops(labels, intensity_image=tubulin_channel)
    
    variances = []

    for prop in props:
        # Filter small artifacts
        if prop.area < 50:
            continue

        # prop.intensity_image gives the intensity values within the bounding box
        # prop.image gives the binary mask within the bounding box
        
        # We need coordinates relative to the centroid of the object
        # prop.coords returns (row, col) coordinates of pixels in the region (global coordinates)
        # prop.centroid returns (row, col) of the centroid (global coordinates)
        
        # Get coordinates of all pixels in the region
        coords = prop.coords # Shape (N, 2)
        
        # Get intensities of these pixels
        # We can extract them from the global image using the coords
        intensities = tubulin_channel[coords[:, 0], coords[:, 1]]
        
        # Total intensity mass
        total_mass = np.sum(intensities)
        
        if total_mass == 0:
            continue

        # Centroid (global)
        cy, cx = prop.weighted_centroid
        
        # Calculate squared distances from weighted centroid for each pixel
        # dist^2 = (y - cy)^2 + (x - cx)^2
        dy = coords[:, 0] - cy
        dx = coords[:, 1] - cx
        squared_distances = dy**2 + dx**2
        
        # Calculate second moment (variance of spatial distribution)
        # Weighted average of squared distances
        # This is effectively the trace of the covariance matrix of the spatial distribution
        # or the "radius of gyration" squared
        spatial_variance = np.sum(squared_distances * intensities) / total_mass
        
        variances.append(spatial_variance)

    if not variances:
        return 0.0

    # Return the mean variance across all detected cells
    return float(np.mean(variances))
