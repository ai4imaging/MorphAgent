def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed
    from skimage.morphology import binary_opening, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) HWC
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected 3-channel image, return 0.0
        return 0.0

    # Channel mapping based on dataset description:
    # Channel 0: Actin (Cell Body)
    # Channel 2: DAPI (Nucleus)
    ch_actin = arr[..., 0]
    ch_dapi = arr[..., 2]

    # Normalize channels to [0, 1] for processing
    def normalize(c):
        vmax = np.percentile(c, 99.5) if c.size > 0 else 1.0
        if vmax <= 0: vmax = 1.0
        return np.clip(c / vmax, 0.0, 1.0)

    actin_norm = normalize(ch_actin)
    dapi_norm = normalize(ch_dapi)

    nuclei_labels = None
    cell_labels = None

    # --- Segmentation Logic ---
    
    # Check if valid segmentation masks are provided
    # We expect masks to be passed as arguments.
    # If provided, we need to distinguish which is nuclei and which is cell.
    # Heuristic: Nuclei masks usually have smaller total foreground area than cell masks.
    
    valid_masks = [m for m in segmentation_masks if m is not None and isinstance(m, np.ndarray)]
    
    if len(valid_masks) >= 2:
        # Sort by total area (number of non-zero pixels)
        # Assuming the two masks correspond to Nuclei and Cells
        # Nuclei < Cells usually
        masks_sorted = sorted(valid_masks, key=lambda x: np.count_nonzero(x))
        
        # The smaller area is likely nuclei, larger is cells
        raw_nuclei_mask = masks_sorted[0]
        raw_cell_mask = masks_sorted[1]

        # Ensure they are labeled (instance segmentation)
        # If the mask is binary (max value == 1 or 255 with few unique values), label it.
        # If it already has many unique values, assume it's an instance mask.
        if len(np.unique(raw_nuclei_mask)) <= 2:
            nuclei_labels = label(raw_nuclei_mask > 0)
        else:
            nuclei_labels = raw_nuclei_mask.astype(int)
            
        if len(np.unique(raw_cell_mask)) <= 2:
            cell_labels = label(raw_cell_mask > 0)
        else:
            cell_labels = raw_cell_mask.astype(int)

    else:
        # --- Fallback: Internal Segmentation Pipeline ---
        
        # 1. Nuclei Segmentation (DAPI)
        # Smooth and threshold
        dapi_smooth = ndimage.gaussian_filter(dapi_norm, sigma=2)
        try:
            thresh_nuc = threshold_otsu(dapi_smooth)
        except:
            thresh_nuc = 0.1
        
        mask_nuc = dapi_smooth > thresh_nuc
        mask_nuc = binary_opening(mask_nuc, footprint=disk(2))
        nuclei_labels = label(mask_nuc)

        # 2. Cell Segmentation (Actin)
        # Smooth and threshold
        actin_smooth = ndimage.gaussian_filter(actin_norm, sigma=2)
        try:
            thresh_cell = threshold_otsu(actin_smooth)
        except:
            thresh_cell = 0.1
            
        mask_cell = actin_smooth > thresh_cell
        
        # Watershed to separate cells based on nuclei markers
        # We use the inverse intensity of actin as the "basin"
        # Markers are the nuclei
        if np.max(nuclei_labels) > 0:
            cell_labels = watershed(-actin_smooth, nuclei_labels, mask=mask_cell)
        else:
            cell_labels = np.zeros_like(mask_cell, dtype=int)

    # --- Feature Computation: Centroid Displacement ---

    if nuclei_labels is None or cell_labels is None or np.max(cell_labels) == 0:
        return 0.0

    # Calculate properties
    props_nuc = regionprops(nuclei_labels)
    props_cell = regionprops(cell_labels)

    # Create a lookup for nuclei centroids
    # If using watershed, label IDs match (Cell 1 contains Nucleus 1).
    # If using external masks, IDs might not match, so we need spatial matching.
    
    # Strategy: Iterate over cells, find the contained nucleus, calculate distance.
    
    displacements = []

    # Map nuclei labels to their centroids for quick access
    nuc_centroids = {p.label: np.array(p.centroid) for p in props_nuc}
    
    # If we generated labels via watershed, the IDs correspond directly.
    # If masks were provided externally, we must check overlap.
    is_watershed_derived = (len(valid_masks) < 2)

    for cell_prop in props_cell:
        c_label = cell_prop.label
        c_centroid = np.array(cell_prop.centroid)
        
        matched_nuc_centroid = None

        if is_watershed_derived:
            # Direct ID matching
            if c_label in nuc_centroids:
                matched_nuc_centroid = nuc_centroids[c_label]
        else:
            # Spatial matching: Find nucleus with max overlap
            # This is computationally more expensive but necessary for unlinked masks
            # Optimization: Check bounding box intersection first
            
            # Extract cell mask
            minr, minc, maxr, maxc = cell_prop.bbox
            cell_submask = (cell_labels[minr:maxr, minc:maxc] == c_label)
            
            # Look at nuclei mask in the same region
            nuc_subcrop = nuclei_labels[minr:maxr, minc:maxc]
            
            # Find nuclei labels inside this cell region
            # We mask the nuclei crop with the binary cell mask to ensure containment
            contained_nuclei = nuc_subcrop[cell_submask]
            
            # Filter out background (0)
            contained_nuclei = contained_nuclei[contained_nuclei > 0]
            
            if contained_nuclei.size > 0:
                # Find the most frequent nucleus label in this cell
                # (The dominant nucleus)
                counts = np.bincount(contained_nuclei)
                dominant_nuc_id = np.argmax(counts)
                
                if dominant_nuc_id in nuc_centroids:
                    matched_nuc_centroid = nuc_centroids[dominant_nuc_id]

        if matched_nuc_centroid is not None:
            # Calculate Euclidean distance
            dist = np.linalg.norm(c_centroid - matched_nuc_centroid)
            displacements.append(dist)

    # --- Aggregation ---
    
    if len(displacements) == 0:
        return 0.0
        
    result = np.mean(displacements)

    return float(result)
