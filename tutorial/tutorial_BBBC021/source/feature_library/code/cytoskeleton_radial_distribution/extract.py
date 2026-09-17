def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed
    from skimage.morphology import binary_dilation, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Channels
    # Channel 0: Actin (Red) - Cytoskeleton
    # Channel 1: Tubulin (Green) - Cytoskeleton
    # Channel 2: DAPI (Blue) - Nucleus
    actin = arr[..., 0]
    tubulin = arr[..., 1]
    dapi = arr[..., 2]

    # Combine Actin and Tubulin for a general "Cytoskeleton" intensity map
    # We take the maximum projection or average to capture structure from both
    cyto_intensity = np.maximum(actin, tubulin)

    # Normalize intensity for weighting
    # Robust max to avoid hot pixels skewing the weights too much
    p99 = np.percentile(cyto_intensity, 99)
    if p99 > 0:
        cyto_intensity = cyto_intensity / p99
    cyto_intensity = np.clip(cyto_intensity, 0, 1)

    # --- Segmentation Logic ---
    # We need:
    # 1. Nuclear Centroids (to measure distance FROM)
    # 2. Cell Masks (to define WHICH pixels belong to which nucleus)

    nuclei_mask = None
    cell_mask = None

    # Check if valid segmentation masks are provided
    # We expect masks to be passed in *segmentation_masks
    # If available, we try to identify which is nuclei and which is cell/cytoplasm
    if len(segmentation_masks) > 0:
        # Simple heuristic: Nuclei masks usually cover less area than cell masks
        # Or we just take the first one if only one is provided
        masks = [np.asarray(m) for m in segmentation_masks if m is not None]
        
        if len(masks) == 1:
            # If only one mask, assume it's nuclei if small objects, or cells if large.
            # For this feature, we critically need separate nuclei to define centers.
            # If we only have one mask, we'll treat it as the object definition
            # and calculate centroids from it.
            nuclei_mask = masks[0]
            cell_mask = masks[0] # Fallback: measure distribution within the provided mask
        elif len(masks) >= 2:
            # Sort by total area (nuclei usually smaller)
            areas = [np.sum(m > 0) for m in masks]
            sorted_indices = np.argsort(areas)
            nuclei_mask = masks[sorted_indices[0]]
            cell_mask = masks[sorted_indices[1]]

    # Fallback: On-the-fly segmentation if no masks provided or they are empty
    if nuclei_mask is None or np.max(nuclei_mask) == 0:
        # Segment Nuclei (DAPI)
        # Smooth slightly
        dapi_smooth = ndimage.gaussian_filter(dapi, sigma=2)
        try:
            thresh_nuc = threshold_otsu(dapi_smooth)
        except ValueError: # Handle empty images
            thresh_nuc = 0
        
        nuclei_bool = dapi_smooth > thresh_nuc
        # Remove small noise
        nuclei_bool = binary_dilation(nuclei_bool, disk(1)) # slight dilation to merge fragments
        nuclei_mask = label(nuclei_bool)

    if cell_mask is None or np.max(cell_mask) == 0:
        # Segment Cells (Watershed)
        # 1. Foreground definition from cytoskeleton
        cyto_smooth = ndimage.gaussian_filter(cyto_intensity, sigma=2)
        try:
            thresh_cyto = threshold_otsu(cyto_smooth)
        except ValueError:
            thresh_cyto = 0
        
        # Make sure cell mask includes nuclei
        foreground = (cyto_smooth > thresh_cyto) | (nuclei_mask > 0)
        
        # 2. Watershed
        # Seeds are the nuclei
        # Basin is the inverted intensity (darker = valleys) or just distance transform
        # Here we use the mask to constrain watershed
        cell_mask = watershed(-cyto_smooth, nuclei_mask, mask=foreground)

    # --- Feature Computation: Intensity-Weighted Radial Distribution ---
    
    # Get properties of nuclei to find centroids
    # We only care about labels present in both masks (ideally they match)
    unique_labels = np.unique(nuclei_mask)
    unique_labels = unique_labels[unique_labels > 0] # remove background

    if len(unique_labels) == 0:
        return 0.0

    # Calculate centroids of nuclei
    # center_of_mass returns (row, col) tuples
    nuc_centroids = ndimage.center_of_mass(dapi, labels=nuclei_mask, index=unique_labels)
    
    # If only one label, center_of_mass returns a tuple, not a list of tuples. Fix this.
    if len(unique_labels) == 1:
        nuc_centroids = [nuc_centroids]

    # Map label ID to centroid
    label_to_centroid = {lbl: cent for lbl, cent in zip(unique_labels, nuc_centroids)}

    # We will iterate over cells in the cell_mask
    # For efficiency with many cells, we can use ndimage.sum and ndimage.mean, 
    # but calculating radial distance requires spatial coordinates.
    
    # Create coordinate grids
    H, W = cell_mask.shape
    y_indices, x_indices = np.indices((H, W))

    weighted_radii = []

    # Iterate through each cell
    # Optimization: Use regionprops on cell_mask to get bounding boxes and slice
    props = regionprops(cell_mask, intensity_image=cyto_intensity)

    for prop in props:
        label_id = prop.label
        
        # Check if we have a corresponding nucleus centroid
        if label_id not in label_to_centroid:
            continue
            
        cy, cx = label_to_centroid[label_id]
        
        # Extract local patch for the cell (bbox)
        min_row, min_col, max_row, max_col = prop.bbox
        
        # Local mask for this specific cell
        local_mask = prop.image # Binary mask of the cell in the bounding box
        local_intensity = prop.intensity_image # Intensity of cytoskeleton in bbox
        
        # Local coordinates
        # prop.coords returns absolute coordinates (row, col)
        # But it's often faster to work with the bbox slices if the cell is dense
        # Let's use the absolute coordinates provided by regionprops for accuracy
        coords = prop.coords # (N, 2) array of (row, col)
        
        if coords.shape[0] == 0:
            continue

        pixel_rows = coords[:, 0]
        pixel_cols = coords[:, 1]
        
        # Get intensities at these coordinates
        # We can't just use local_intensity because it's a rectangular slice including 0s for background
        # We need the values specifically at the mask pixels.
        # regionprops.intensity_image[local_mask] gives the values inside the mask
        pixel_intensities = local_intensity[local_mask]
        
        # Calculate Euclidean distance from nuclear centroid to each pixel
        # dist = sqrt((r - cy)^2 + (c - cx)^2)
        dist_sq = (pixel_rows - cy)**2 + (pixel_cols - cx)**2
        distances = np.sqrt(dist_sq)
        
        # Compute Intensity Weighted Mean Radius
        # Sum(Distance * Intensity) / Sum(Intensity)
        total_intensity = np.sum(pixel_intensities)
        
        if total_intensity > 1e-6:
            weighted_mean_radius = np.sum(distances * pixel_intensities) / total_intensity
            weighted_radii.append(weighted_mean_radius)
        else:
            # If intensity is effectively zero, the radius is undefined or 0
            weighted_radii.append(0.0)

    # Aggregate results
    if not weighted_radii:
        return 0.0
    
    # Return the median of the population to be robust against segmentation errors
    result = np.median(weighted_radii)

    return float(result)
