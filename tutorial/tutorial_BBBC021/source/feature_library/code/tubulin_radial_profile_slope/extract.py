def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from scipy import stats
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_dilation, disk

    # --- 1. Data Loading and Preprocessing ---
    # Convert to float32 for processing
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality and extract channels
    # Expected shape: (512, 512, 3)
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green) - TARGET CHANNEL
    # Channel 2: DAPI (Blue) - REFERENCE CHANNEL (Nucleus)
    
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract relevant channels
    tubulin_ch = arr[:, :, 1]
    dapi_ch = arr[:, :, 2]
    
    # Normalize intensity to [0, 1]
    # The dataset is uint8 (0-255), so simple division is robust
    tubulin_norm = tubulin_ch / 255.0
    dapi_norm = dapi_ch / 255.0

    # --- 2. Segmentation Logic ---
    # We need to identify individual cells and their nuclei to compute radial profiles.
    
    nuclei_labels = None
    cell_labels = None

    # Check if segmentation masks are provided via arguments
    if len(segmentation_masks) > 0:
        # Heuristic: usually nuclei is smaller/first, cells larger/second, 
        # or they might be passed in specific order. 
        # Without strict metadata, we assume:
        # If 1 mask: it's likely nuclei or cells. We need both for accurate radial profiling.
        # If 2 masks: likely nuclei and cells.
        
        # Let's try to identify which is which based on coverage or assume order if standard.
        # Standard convention often puts nuclei first or we can infer from size.
        # However, to be robust, if we have masks, we use them.
        
        # If we have at least one mask, let's assume the first one is nuclei or cells.
        # A common strategy for radial profiling is:
        # 1. Identify nucleus center.
        # 2. Identify cell boundary.
        
        mask1 = segmentation_masks[0]
        if len(segmentation_masks) >= 2:
            mask2 = segmentation_masks[1]
            # Heuristic: Nuclei usually cover less area than cells
            if np.sum(mask1 > 0) < np.sum(mask2 > 0):
                nuclei_labels = mask1
                cell_labels = mask2
            else:
                nuclei_labels = mask2
                cell_labels = mask1
        else:
            # Only one mask. If it's nuclei, we need to estimate cell boundaries.
            # If it's cells, we need to estimate nuclei centers.
            # Let's assume it's a general object mask and try to split it or use it as cell mask.
            # For this specific feature, accurate nuclei centers are critical.
            # We will fallback to computing our own nuclei from DAPI if only 1 mask is given 
            # to ensure we have the "center" correct, and use the mask as the "cell".
            cell_labels = mask1
            # We will generate nuclei_labels below in the fallback block if it's None

    # --- 3. Fallback Segmentation (if masks missing) ---
    if nuclei_labels is None:
        # Otsu threshold on DAPI for nuclei
        try:
            thresh_nuc = threshold_otsu(dapi_norm)
        except:
            thresh_nuc = 0.1
        nuclei_mask = dapi_norm > thresh_nuc
        # Clean up noise
        nuclei_mask = binary_dilation(nuclei_mask, disk(1)) # slight smoothing
        nuclei_labels = label(nuclei_mask)

    if cell_labels is None:
        # Watershed approach to define cell boundaries based on nuclei seeds
        # Combine channels to find cell body
        combined = (tubulin_norm + dapi_norm) / 2.0
        try:
            thresh_cell = threshold_otsu(combined)
        except:
            thresh_cell = 0.05
        
        cell_mask = combined > thresh_cell
        
        # Use distance transform for watershed
        # We want to expand from nuclei to cell boundaries
        # Distance from background
        distance = ndimage.distance_transform_edt(cell_mask)
        
        # We need markers for watershed. The nuclei_labels are perfect markers.
        # However, nuclei_labels might not be strictly inside cell_mask due to threshold differences.
        # We ensure markers intersect with mask.
        markers = nuclei_labels
        
        # Watershed
        # In skimage watershed, we flood 'image' starting from 'markers'.
        # We use negative distance as the topographic surface (basins at peaks).
        from skimage.segmentation import watershed
        cell_labels = watershed(-distance, markers, mask=cell_mask)

    # --- 4. Feature Computation: Radial Profile Slope ---
    
    props_nuclei = regionprops(nuclei_labels)
    # Map label ID to regionprop object for quick access
    nuclei_map = {p.label: p for p in props_nuclei}
    
    props_cells = regionprops(cell_labels, intensity_image=tubulin_norm)
    
    slopes = []

    for cell_prop in props_cells:
        label_id = cell_prop.label
        
        # We need the corresponding nucleus to find the center
        if label_id not in nuclei_map:
            continue
            
        nuc_prop = nuclei_map[label_id]
        
        # Centroid of the nucleus (row, col)
        cy, cx = nuc_prop.centroid
        
        # Get coordinates of all pixels in the cell
        # regionprops.coords returns (row, col) for every pixel in the region
        cell_coords = cell_prop.coords # Shape (N, 2)
        
        if len(cell_coords) < 50: # Skip tiny artifacts
            continue
            
        # Extract pixel intensities for the cell
        # We can use the intensity_image from regionprops, but coords are global.
        # It's easier to index the global image.
        pixel_intensities = tubulin_norm[cell_coords[:, 0], cell_coords[:, 1]]
        
        # Calculate Euclidean distance from nucleus centroid to each pixel
        # dist = sqrt((r - cy)^2 + (c - cx)^2)
        dy = cell_coords[:, 0] - cy
        dx = cell_coords[:, 1] - cx
        distances = np.sqrt(dy**2 + dx**2)
        
        # Normalize distances to [0, 1] for this cell
        # This makes the profile scale-invariant (independent of cell size)
        max_dist = np.max(distances)
        if max_dist == 0:
            continue
        
        norm_distances = distances / max_dist
        
        # Binning the profile
        # We want to compute the mean intensity at radial intervals.
        # Instead of complex binning, we can just do a linear regression on all points
        # or bin them first to reduce noise. Binning is usually more robust for "profile" features.
        
        num_bins = 20
        bins = np.linspace(0, 1.0, num_bins + 1)
        
        # Digitise distances to find which bin they belong to
        bin_indices = np.digitize(norm_distances, bins) - 1
        
        bin_means = []
        bin_centers = []
        
        for i in range(num_bins):
            # Select pixels in this bin
            mask_bin = bin_indices == i
            if np.any(mask_bin):
                mean_val = np.mean(pixel_intensities[mask_bin])
                bin_means.append(mean_val)
                # Center of the bin
                bin_centers.append((bins[i] + bins[i+1]) / 2.0)
        
        if len(bin_means) < 3: # Need at least a few points for a slope
            continue
            
        # Linear Regression
        # X: Normalized Radius (bin centers)
        # Y: Mean Intensity
        slope, intercept, r_value, p_value, std_err = stats.linregress(bin_centers, bin_means)
        
        # Check for NaN
        if not np.isnan(slope):
            slopes.append(slope)

    # --- 5. Aggregation ---
    # Return the median slope across all cells in the image.
    # Median is robust to segmentation errors.
    if not slopes:
        return 0.0
        
    result = np.median(slopes)
    
    return float(result)
