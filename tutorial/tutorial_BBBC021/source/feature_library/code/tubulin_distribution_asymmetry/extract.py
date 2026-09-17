def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import clear_border, watershed
    from skimage.morphology import dilation, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality and validate input
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Channels
    # Channel 1: Tubulin (Green) - The signal distribution we are measuring
    # Channel 2: DAPI (Blue) - The reference point (Nucleus)
    tubulin_ch = arr[..., 1]
    nuclei_ch = arr[..., 2]

    # Normalize Intensity (0-1 range)
    # Robust normalization using percentiles to handle outliers
    def normalize_channel(ch):
        p99 = np.percentile(ch, 99.5)
        if p99 > 0:
            ch = ch / p99
        return np.clip(ch, 0.0, 1.0)

    tubulin_norm = normalize_channel(tubulin_ch)
    nuclei_norm = normalize_channel(nuclei_ch)

    # --- Segmentation Logic ---
    # We need two masks:
    # 1. Nuclei Mask: To define the reference center (Nuclear Centroid).
    # 2. Cell Mask: To define the boundary of the cytoplasm where Tubulin exists.
    
    nuclei_mask = None
    cell_mask = None

    # Check provided masks
    if len(segmentation_masks) > 0:
        # Heuristic: If 1 mask, assume it's nuclei (common in this dataset context).
        # If 2 masks, assume [nuclei, cells] or [cells, nuclei].
        # We usually check overlap with channels to confirm, but here we'll try to use the first valid one as nuclei.
        
        # Try to identify which mask is which based on typical ordering or just use the first available
        # In many pipelines, mask 0 is nuclei, mask 1 is cells.
        m0 = segmentation_masks[0]
        if m0 is not None and m0.shape == arr.shape[:2]:
            nuclei_mask = m0
        
        if len(segmentation_masks) > 1:
            m1 = segmentation_masks[1]
            if m1 is not None and m1.shape == arr.shape[:2]:
                cell_mask = m1

    # Fallback Segmentation if masks are missing
    if nuclei_mask is None:
        try:
            thresh_n = threshold_otsu(nuclei_norm)
            nuclei_mask = label(nuclei_norm > thresh_n)
            nuclei_mask = clear_border(nuclei_mask)
        except:
            return 0.0

    if cell_mask is None:
        # Create a cell mask by expanding nuclei (Voronoi-like watershed)
        # This associates cytoplasmic tubulin with the nearest nucleus
        try:
            # Threshold tubulin to find general cell foreground
            thresh_t = threshold_otsu(tubulin_norm)
            foreground = (tubulin_norm > thresh_t) | (nuclei_mask > 0)
            
            # Watershed: seeds are nuclei, mask is foreground
            # We use the inverse of tubulin intensity as the topographic map to guide boundaries slightly
            cell_mask = watershed(-tubulin_norm, nuclei_mask, mask=foreground)
        except:
            # If watershed fails, just use dilated nuclei as a proxy for cell area
            cell_mask = dilation(nuclei_mask, disk(5))

    # Ensure masks are labeled integers
    nuclei_labels = nuclei_mask.astype(np.int32)
    cell_labels = cell_mask.astype(np.int32)

    # Get unique cell IDs (excluding background 0)
    unique_cells = np.unique(cell_labels)
    if len(unique_cells) <= 1:
        return 0.0

    # --- Feature Computation: Tubulin Asymmetry ---
    # Logic:
    # 1. For each cell, find the Geometric Centroid of the Nucleus (Reference Point).
    # 2. Find the Intensity-Weighted Center of Mass of Tubulin within that cell (Signal Center).
    # 3. Calculate Euclidean distance between Reference Point and Signal Center.
    # 4. Normalize by cell size (radius) to make it scale-invariant.

    # Pre-calculate centers of mass
    # We need:
    # A. Center of mass of the NUCLEUS mask (binary) -> Geometric center of nucleus
    # B. Center of mass of the TUBULIN intensity (weighted) inside the CELL mask
    
    # Note: ndimage.center_of_mass returns list of tuples [(r, c), ...]
    # The 'index' argument allows computing for all labels at once.
    
    # Filter valid indices (present in both masks)
    # We iterate through regionprops to ensure we match nucleus to cell correctly
    # Assumption: The label ID in nuclei_mask corresponds to the same label ID in cell_mask
    # (This is true for watershed expansion, and usually true for matched segmentation pipelines)
    
    props_nuclei = regionprops(nuclei_labels)
    
    asymmetry_scores = []

    for prop in props_nuclei:
        label_id = prop.label
        
        # 1. Get Nuclear Centroid (Geometric)
        # regionprops.centroid returns (row, col)
        nuc_centroid = np.array(prop.centroid)
        
        # 2. Get Tubulin Center of Mass (Weighted)
        # We need to isolate the tubulin pixels belonging to this specific cell label
        # Optimization: Use the bounding box from the cell mask if possible, but here we rely on global mask for simplicity in code
        # To avoid processing the whole 512x512 array for every cell, we use a slice based on the nucleus bbox + margin, 
        # or just mask the specific region if the cell is large.
        
        # Let's verify if this label exists in the cell mask
        # (It should if cell_mask was derived from nuclei_mask)
        
        # Extract slices to speed up processing
        minr, minc, maxr, maxc = prop.bbox
        # Expand bbox slightly to capture cytoplasm (heuristic expansion)
        pad = 60
        minr = max(0, minr - pad)
        minc = max(0, minc - pad)
        maxr = min(arr.shape[0], maxr + pad)
        maxc = min(arr.shape[1], maxc + pad)
        
        # Crop arrays
        sub_cell_labels = cell_labels[minr:maxr, minc:maxc]
        sub_tubulin = tubulin_norm[minr:maxr, minc:maxc]
        
        # Create binary mask for this specific cell in the crop
        current_cell_mask = (sub_cell_labels == label_id)
        
        if not np.any(current_cell_mask):
            continue
            
        # Calculate weighted center of mass for tubulin
        # If tubulin is completely dark, center_of_mass might return NaN or tuple of NaNs
        try:
            tub_com_local = ndimage.center_of_mass(sub_tubulin, labels=current_cell_mask, index=1)
            if np.any(np.isnan(tub_com_local)):
                continue
            
            # Convert local crop coordinates back to global coordinates
            tub_centroid = np.array(tub_com_local) + np.array([minr, minc])
            
            # 3. Calculate Displacement
            displacement = np.linalg.norm(tub_centroid - nuc_centroid)
            
            # 4. Normalize by Cell Size
            # We use the equivalent radius of the cell area (sqrt(Area/pi))
            # Area is the count of pixels in the cell mask
            cell_area = np.sum(current_cell_mask)
            if cell_area == 0:
                continue
                
            equivalent_radius = np.sqrt(cell_area / np.pi)
            
            # Avoid division by zero
            if equivalent_radius < 1.0:
                norm_score = 0.0
            else:
                norm_score = displacement / equivalent_radius
                
            asymmetry_scores.append(norm_score)
            
        except Exception:
            continue

    # Aggregation
    if not asymmetry_scores:
        return 0.0

    # Return the mean asymmetry across the population
    return float(np.mean(asymmetry_scores))
