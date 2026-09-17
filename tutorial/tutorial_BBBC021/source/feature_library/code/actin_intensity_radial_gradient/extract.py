def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from scipy import stats
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed, clear_border
    from skimage.morphology import closing, square

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), uint8
    # Channel 0: Actin (Red) - Target for intensity measurement
    # Channel 1: Tubulin (Green)
    # Channel 2: DAPI (Blue) - Target for center definition
    
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected 3-channel image, return 0.0
        return 0.0

    # Extract relevant channels
    actin_channel = arr[..., 0]
    dapi_channel = arr[..., 2]

    # Intensity normalization (0-1 range)
    # Normalize Actin for intensity measurements
    vmax_actin = np.percentile(actin_channel, 99.5) if actin_channel.size > 0 else 1.0
    if vmax_actin > 0:
        actin_norm = actin_channel / vmax_actin
    else:
        actin_norm = actin_channel
    actin_norm = np.clip(actin_norm, 0.0, 1.0)

    # Normalize DAPI for segmentation
    vmax_dapi = np.percentile(dapi_channel, 99.5) if dapi_channel.size > 0 else 1.0
    if vmax_dapi > 0:
        dapi_norm = dapi_channel / vmax_dapi
    else:
        dapi_norm = dapi_channel
    dapi_norm = np.clip(dapi_norm, 0.0, 1.0)

    # --- Segmentation Logic ---
    # We need instance segmentation to define "Center" (Nucleus) and "Periphery" (Cell Boundary) per cell.
    
    cell_labels = None
    nuclei_labels = None

    # Check if valid segmentation masks are provided
    # We expect masks that might separate nuclei and cells, but the interface is generic.
    # If masks are provided, we try to use them.
    if len(segmentation_masks) > 0:
        # Heuristic: If multiple masks, assume one is nuclei and one is cell based on size or overlap
        # For simplicity in this generic function, if masks exist, we assume the first one is a cell/nuclei instance mask.
        # Ideally, we need separate nuclei and cell masks.
        # If only one mask is provided, we treat it as the cell mask and infer nuclei centers from DAPI within it.
        
        # Let's try to identify nuclei vs cell mask if 2 are provided
        if len(segmentation_masks) >= 2:
            mask1 = segmentation_masks[0]
            mask2 = segmentation_masks[1]
            # Usually nuclei are smaller.
            if np.sum(mask1 > 0) < np.sum(mask2 > 0):
                nuclei_labels = mask1
                cell_labels = mask2
            else:
                nuclei_labels = mask2
                cell_labels = mask1
        else:
            # Only one mask, assume it's the cell mask (cytoplasm+nucleus)
            cell_labels = segmentation_masks[0]
            # We will derive nuclei centers from the DAPI channel inside these cell labels later
    
    # Fallback: Internal Segmentation if no masks or invalid masks
    if cell_labels is None:
        # 1. Detect Nuclei (Seeds)
        try:
            thresh_nuc = threshold_otsu(dapi_norm)
        except:
            thresh_nuc = 0.1
        
        nuclei_mask = dapi_norm > thresh_nuc
        nuclei_mask = closing(nuclei_mask, square(3))
        nuclei_labels = label(nuclei_mask)
        
        # 2. Detect Cell Body (Foreground)
        # Use Actin channel for cell body
        try:
            thresh_cell = threshold_otsu(actin_norm)
        except:
            thresh_cell = 0.1
        
        # Lower threshold slightly to capture faint edges
        cell_mask = actin_norm > (thresh_cell * 0.8)
        cell_mask = closing(cell_mask, square(3))
        
        # 3. Watershed to separate cells
        # Use nuclei as markers
        if np.max(nuclei_labels) > 0:
            # Distance transform for watershed topology (optional, but intensity often works for fluorescence)
            # Here we use inverse intensity of actin as the "basin"
            cell_labels = watershed(-actin_norm, nuclei_labels, mask=cell_mask)
        else:
            cell_labels = label(cell_mask) # Fallback to connected components if no nuclei found

    # Remove border cells to avoid artifacts in radial distribution
    cell_labels = clear_border(cell_labels)
    
    # If still no objects, return 0.0
    if cell_labels.max() == 0:
        return 0.0

    # --- Feature Computation: Radial Gradient ---
    
    gradients = []
    
    # Iterate over each cell
    props = regionprops(cell_labels)
    
    for prop in props:
        # Optimization: Work on the bounding box of the cell
        minr, minc, maxr, maxc = prop.bbox
        
        # Extract cell mask and intensity in the bounding box
        cell_mask_crop = cell_labels[minr:maxr, minc:maxc] == prop.label
        actin_crop = actin_norm[minr:maxr, minc:maxc]
        
        if np.sum(cell_mask_crop) < 50: # Skip tiny fragments
            continue

        # Determine the "Center" of this cell
        # If we have a nuclei mask, find the nucleus centroid within this cell region
        # Otherwise, use the weighted centroid of the DAPI signal within the cell mask
        
        dapi_crop = dapi_norm[minr:maxr, minc:maxc]
        
        # Calculate centroid of DAPI signal within the cell mask
        # We mask DAPI with the cell shape to ensure we only look inside this specific cell
        masked_dapi = dapi_crop * cell_mask_crop
        
        if np.sum(masked_dapi) == 0:
            # Fallback to geometric center of the cell mask if no DAPI signal
            cy_local, cx_local = ndimage.center_of_mass(cell_mask_crop)
        else:
            # Weighted center of mass of DAPI (Nucleus center)
            cy_local, cx_local = ndimage.center_of_mass(masked_dapi)
            
        # Create coordinate grids relative to the center
        h, w = cell_mask_crop.shape
        y_indices, x_indices = np.indices((h, w))
        
        # Calculate Euclidean distance from the center for every pixel
        distances = np.sqrt((y_indices - cy_local)**2 + (x_indices - cx_local)**2)
        
        # Normalize distance: 0 at center, 1 at the furthest point in the cell mask
        # Alternatively, we can use the distance to the boundary to normalize:
        # Normalized Radius = Dist_from_Center / (Dist_from_Center + Dist_to_Edge)
        # This handles irregular shapes better than simple max-normalization.
        
        # Distance transform from the background (gives distance to nearest edge)
        # We need the distance inside the mask.
        dist_to_edge = ndimage.distance_transform_edt(cell_mask_crop)
        
        # Avoid division by zero
        denominator = distances + dist_to_edge
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_radius = distances / denominator
            normalized_radius[denominator == 0] = 0
            
        # Flatten arrays to select only pixels within the cell mask
        mask_flat = cell_mask_crop.ravel()
        intensities = actin_crop.ravel()[mask_flat]
        radii = normalized_radius.ravel()[mask_flat]
        
        if len(intensities) < 10:
            continue
            
        # Calculate the gradient (slope) of Intensity vs Normalized Radius
        # We expect:
        # Positive slope -> Higher intensity at edges (Cortical)
        # Negative slope -> Higher intensity at center (Perinuclear)
        # Zero slope -> Diffuse
        
        # Use linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(radii, intensities)
        
        if not np.isnan(slope):
            gradients.append(slope)

    # Aggregate results
    if not gradients:
        return 0.0
        
    # Return the median gradient across all cells to be robust to outliers
    result = np.median(gradients)
    
    return float(result)
