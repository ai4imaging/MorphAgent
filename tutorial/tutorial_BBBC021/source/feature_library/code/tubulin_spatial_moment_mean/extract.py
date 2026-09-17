def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    # 1. Input Handling and Preprocessing
    # Ensure image is float32 for calculations
    img_arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensionality
    if img_arr.ndim != 3:
        return 0.0
    
    # Channel Mapping based on dataset description:
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green) - TARGET for intensity
    # Channel 2: DAPI (Blue) - TARGET for nuclear centroid
    
    # Extract channels
    try:
        tubulin_ch = img_arr[..., 1]
        dapi_ch = img_arr[..., 2]
    except IndexError:
        return 0.0

    # Normalize Tubulin channel for weighting (avoid huge numbers, though float32 handles it)
    # Simple min-max to 0-1 range
    t_min, t_max = tubulin_ch.min(), tubulin_ch.max()
    if t_max > t_min:
        tubulin_norm = (tubulin_ch - t_min) / (t_max - t_min)
    else:
        tubulin_norm = np.zeros_like(tubulin_ch)

    # 2. Segmentation Logic
    # We need two things:
    # A. Nuclear Centroids (Reference points)
    # B. Cell Masks (Boundaries for Tubulin pixels belonging to that nucleus)

    nuclei_mask = None
    cell_mask = None

    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Heuristic to identify masks. Usually, nuclei are smaller/more numerous or labeled specifically.
        # Without specific filenames, we assume order or analyze content.
        # Common convention: often nuclei is first or separate.
        # Let's try to identify based on overlap with DAPI vs Tubulin if possible, or just assume:
        # If 2 masks: likely [cell_mask, nuclei_mask] or [nuclei_mask, cell_mask].
        # Given the prompt doesn't specify order, we'll implement a fallback generation 
        # if the provided masks aren't usable, but let's try to use them.
        
        # Strategy: If masks are provided, we assume the one with better overlap with DAPI is nuclei
        # and the one covering more area is cells.
        
        masks = [np.array(m) for m in segmentation_masks]
        
        if len(masks) == 1:
            # If only one mask, assume it's cell boundaries. We still need nuclei centroids.
            # We will compute nuclei centroids from DAPI within these cell regions.
            cell_mask = masks[0]
        elif len(masks) >= 2:
            # Determine which is which by area
            area_0 = np.sum(masks[0] > 0)
            area_1 = np.sum(masks[1] > 0)
            if area_0 > area_1:
                cell_mask = masks[0]
                nuclei_mask = masks[1]
            else:
                cell_mask = masks[1]
                nuclei_mask = masks[0]

    # Fallback / Refinement if masks are missing
    if nuclei_mask is None:
        # Generate nuclei mask from DAPI
        # Smooth DAPI
        dapi_smooth = ndimage.gaussian_filter(dapi_ch, sigma=2)
        try:
            thresh = threshold_otsu(dapi_smooth)
            nuclei_bin = dapi_smooth > thresh
            # Label nuclei
            nuclei_mask = label(nuclei_bin)
        except:
            return 0.0

    if cell_mask is None:
        # Generate cell mask using watershed on Tubulin seeded by Nuclei
        # Smooth Tubulin
        tub_smooth = ndimage.gaussian_filter(tubulin_ch, sigma=2)
        try:
            thresh_tub = threshold_otsu(tub_smooth) if tub_smooth.max() > tub_smooth.min() else 0
            cell_bin = tub_smooth > thresh_tub
            # Watershed
            # Distance transform for background
            # Use nuclei as markers
            cell_mask = watershed(-tub_smooth, nuclei_mask, mask=cell_bin)
        except:
            # If watershed fails, just use nuclei mask as cell mask (minimal fallback)
            cell_mask = nuclei_mask

    # 3. Feature Computation: Spatial Moment
    # Formula: Sum(Intensity_i * Distance_i) / Sum(Intensity_i)
    # Distance_i is distance from pixel i to its nucleus centroid.
    
    # Get properties of nuclei to find centroids quickly
    # We need to map cell_label -> nucleus_centroid
    
    # If nuclei_mask and cell_mask are distinct (e.g. from separate files), labels might not match.
    # We need to link them.
    
    # Get all cell labels
    cell_labels = np.unique(cell_mask)
    cell_labels = cell_labels[cell_labels > 0]
    
    spatial_moments = []

    # Pre-calculate nuclei centroids
    # We iterate over nuclei mask regionprops
    nuclei_props = regionprops(nuclei_mask)
    # Create a spatial index or map for nuclei centroids? 
    # Simpler: For each cell, find the nucleus inside it.
    
    # Optimization: Use bounding boxes from cell_mask to process locally
    cell_props = regionprops(cell_mask, intensity_image=tubulin_norm)
    
    for cell_prop in cell_props:
        # Get the label of the current cell
        c_label = cell_prop.label
        
        # Extract the bounding box of the cell
        minr, minc, maxr, maxc = cell_prop.bbox
        
        # Extract the sub-masks and sub-images
        cell_submask = cell_mask[minr:maxr, minc:maxc] == c_label
        nuclei_submask = nuclei_mask[minr:maxr, minc:maxc]
        
        # Find the nucleus associated with this cell
        # We look for the most frequent non-zero nucleus label inside the cell region
        nuclei_in_cell = nuclei_submask[cell_submask]
        nuclei_in_cell = nuclei_in_cell[nuclei_in_cell > 0]
        
        if len(nuclei_in_cell) == 0:
            continue # No nucleus found in this cell
            
        # Get the dominant nucleus label
        n_label = np.bincount(nuclei_in_cell).argmax()
        
        # Find centroid of this nucleus
        # We can calculate it directly from the submask to be fast, or look it up if we precalculated
        # Let's calculate locally relative to the slice, then adjust
        n_coords = np.argwhere(nuclei_submask == n_label)
        if len(n_coords) == 0:
            continue
            
        # Local centroid (row, col)
        n_centroid_local = n_coords.mean(axis=0)
        
        # Get coordinates of all pixels in the cell
        # cell_prop.coords returns global coordinates, let's use local for speed
        cell_pixel_coords_local = np.argwhere(cell_submask)
        
        # Get intensities of these pixels
        # cell_prop.image is the binary mask, cell_prop.intensity_image is the intensity
        intensities = cell_prop.intensity_image[cell_submask]
        
        # Calculate distances
        # dist = sqrt((r - r0)^2 + (c - c0)^2)
        diffs = cell_pixel_coords_local - n_centroid_local
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        
        # Weighted sum
        total_intensity = np.sum(intensities)
        
        if total_intensity > 0:
            weighted_dist = np.sum(intensities * distances)
            moment = weighted_dist / total_intensity
            spatial_moments.append(moment)

    # 4. Aggregation
    if not spatial_moments:
        return 0.0
        
    result = np.mean(spatial_moments)
    
    return float(result)
