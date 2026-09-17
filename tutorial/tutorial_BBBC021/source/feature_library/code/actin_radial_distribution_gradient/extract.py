def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage, stats
    from skimage import filters, measure, morphology, segmentation

    # Convert to appropriate array type and handle dimensions
    # Dataset is (512, 512, 3), uint8
    img = np.asarray(img)
    
    # Basic validation
    if img.ndim != 3 or img.shape[2] != 3:
        return 0.0

    # Normalize image to [0, 1]
    # Using percentile for robust max to avoid hot pixel issues
    img_float = img.astype(np.float32)
    vmax = np.percentile(img_float, 99.5) if img_float.size > 0 else 255.0
    if vmax == 0: vmax = 1.0
    img_norm = img_float / vmax
    img_norm = np.clip(img_norm, 0.0, 1.0)

    # Extract Channels based on dataset description
    # Channel 0: Actin (Target for distribution)
    # Channel 1: Tubulin
    # Channel 2: DAPI (Nucleus - Center reference)
    actin_ch = img_norm[:, :, 0]
    dapi_ch = img_norm[:, :, 2]
    
    # --- Segmentation Logic ---
    # We perform on-the-fly segmentation to ensure we have matched nuclei and cell boundaries
    # derived directly from the provided channels.
    
    # 1. Detect Nuclei (Seeds)
    try:
        thresh_nuc = filters.threshold_otsu(dapi_ch)
    except ValueError: # Handle empty images
        return 0.0
        
    mask_nuc = dapi_ch > thresh_nuc
    # Clean up nuclei
    mask_nuc = morphology.remove_small_objects(mask_nuc, min_size=20)
    mask_nuc = morphology.binary_closing(mask_nuc, morphology.disk(2))
    # Label nuclei
    markers_nuc = measure.label(mask_nuc)
    
    # 2. Detect Cell Foreground (Mask)
    # Combine Actin and Tubulin (Ch1) for better cell body definition, or just use Actin
    # Using Actin primarily as it's the target, but adding Tubulin helps define the boundary
    tubulin_ch = img_norm[:, :, 1]
    cell_signal = actin_ch + tubulin_ch
    try:
        thresh_cell = filters.threshold_otsu(cell_signal)
    except ValueError:
        thresh_cell = 0.1
    
    mask_cell = cell_signal > thresh_cell
    # Ensure nuclei are part of the cell mask
    mask_cell = np.logical_or(mask_cell, mask_nuc)
    
    # 3. Watershed to define cell territories
    # We use the gradient of the image as the "elevation" map, seeded by nuclei
    # This partitions the space among nuclei
    elevation_map = filters.sobel(cell_signal)
    cell_labels = segmentation.watershed(elevation_map, markers_nuc, mask=mask_cell)
    
    # --- Feature Extraction: Radial Distribution Gradient ---
    
    slopes = []
    
    # Get properties of labeled regions
    # We need to iterate to process each cell individually
    props = measure.regionprops(cell_labels, intensity_image=actin_ch)
    
    for prop in props:
        # Skip very small artifacts
        if prop.area < 100:
            continue
            
        # 1. Define Center
        # Ideally, we use the centroid of the nucleus *within* this cell region.
        # Since markers_nuc was used as seeds, the label 'prop.label' in markers_nuc
        # corresponds to the nucleus for this cell.
        
        # Extract the bounding box slice to work on smaller arrays
        minr, minc, maxr, maxc = prop.bbox
        
        # Local masks
        local_cell_mask = cell_labels[minr:maxr, minc:maxc] == prop.label
        local_nuc_mask = markers_nuc[minr:maxr, minc:maxc] == prop.label
        
        # If nucleus mask is lost (rare), fall back to cell centroid
        if np.sum(local_nuc_mask) > 0:
            # Calculate centroid of the nucleus mask
            coords_nuc = np.argwhere(local_nuc_mask)
            center_y, center_x = np.mean(coords_nuc, axis=0)
        else:
            # Fallback to local centroid of the cell
            center_y, center_x = prop.local_centroid
            
        # 2. Calculate Radial Distances
        # Get coordinates of all pixels in the cell
        coords_cell = np.argwhere(local_cell_mask)
        if coords_cell.shape[0] == 0:
            continue
            
        # Euclidean distance from center
        distances = np.sqrt((coords_cell[:, 0] - center_y)**2 + (coords_cell[:, 1] - center_x)**2)
        
        # Normalize distances (0 at center, 1 at furthest edge)
        max_dist = np.max(distances)
        if max_dist == 0:
            continue
        norm_distances = distances / max_dist
        
        # 3. Get Intensities
        # Extract actin intensities for these pixels
        # prop.image is the binary mask, prop.intensity_image is the actin crop
        # We need to index into the intensity image using the local mask
        local_actin = prop.intensity_image
        intensities = local_actin[local_cell_mask]
        
        # 4. Binning and Regression
        # Direct pixel regression is noisy. We bin the radius.
        num_bins = 10
        bins = np.linspace(0, 1, num_bins + 1)
        
        # Compute mean intensity per bin
        bin_means = []
        bin_centers = []
        
        # Digitizing tells us which bin each pixel falls into
        # indices 1 to 10
        digitized = np.digitize(norm_distances, bins)
        
        valid_bins = 0
        for i in range(1, num_bins + 1):
            mask_bin = (digitized == i)
            if np.any(mask_bin):
                mean_val = np.mean(intensities[mask_bin])
                bin_means.append(mean_val)
                # Center of the bin
                bin_centers.append((bins[i-1] + bins[i]) / 2.0)
                valid_bins += 1
        
        # Need at least 2 points for a line
        if valid_bins < 2:
            continue
            
        # Linear Regression
        # x = radius (normalized), y = intensity
        slope, intercept, r_value, p_value, std_err = stats.linregress(bin_centers, bin_means)
        
        # Check for NaN slopes
        if not np.isnan(slope):
            slopes.append(slope)

    # --- Aggregation ---
    # Return the median slope across all cells
    # Positive slope: Cortical actin (higher at edge)
    # Negative slope: Perinuclear actin (higher at center)
    if not slopes:
        return 0.0
        
    result = np.median(slopes)
    
    return float(result)
