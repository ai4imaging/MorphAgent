def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk
    from skimage.segmentation import watershed, clear_border

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality and validate channels
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return 0.0

    # Extract relevant channels
    # Channel 0: Actin (Red) -> Cytoskeleton / Cell Body
    # Channel 2: DAPI (Blue) -> Nucleus
    actin_ch = arr[:, :, 0]
    dapi_ch = arr[:, :, 2]

    # Normalize channels to [0, 1] for processing
    def normalize(ch):
        vmax = np.percentile(ch, 99.5) if ch.size > 0 else 1.0
        if vmax <= 0: vmax = 1.0
        return np.clip(ch / vmax, 0.0, 1.0)

    actin_norm = normalize(actin_ch)
    dapi_norm = normalize(dapi_ch)

    # --- Segmentation Logic ---
    # We need instance segmentation to link specific nuclei to specific cell bodies.
    # If masks are provided, we try to use them. If not, we generate them.
    
    nuclei_labels = None
    cell_labels = None

    # Check if valid segmentation masks are provided
    # We expect at least one mask if provided. 
    # Usually, if masks are provided, they might be (nuclei, cells) or just cells.
    # Given the variability, we will implement a robust fallback:
    # 1. If masks exist, try to identify which is which or use the first one as cell labels.
    # 2. If no masks, perform Otsu + Watershed segmentation.
    
    # For this specific feature (Nucleus-Actin offset), precise instance matching is crucial.
    # Often, provided masks might not perfectly align instances if they come from different files.
    # To ensure high fidelity for this specific geometric calculation, we will prioritize
    # a fresh segmentation pipeline derived from the input image itself, as this guarantees
    # the nucleus label 'k' corresponds to the cell body label 'k'.
    
    # --- Robust On-the-fly Segmentation ---
    
    # 1. Segment Nuclei (Seeds)
    try:
        # Gaussian blur to reduce noise
        dapi_smooth = ndimage.gaussian_filter(dapi_norm, sigma=2)
        thresh_nuc = threshold_otsu(dapi_smooth)
        mask_nuc = dapi_smooth > thresh_nuc
        # Clean up
        mask_nuc = binary_opening(mask_nuc, footprint=disk(2))
        # Label nuclei
        nuclei_labels = label(mask_nuc)
        # Remove border artifacts which distort centroids
        nuclei_labels = clear_border(nuclei_labels)
    except Exception:
        return 0.0

    if nuclei_labels.max() == 0:
        return 0.0

    # 2. Segment Cell Bodies (Watershed)
    try:
        # Gaussian blur for actin
        actin_smooth = ndimage.gaussian_filter(actin_norm, sigma=2)
        
        # Determine background mask
        try:
            thresh_actin = threshold_otsu(actin_smooth)
        except:
            thresh_actin = 0.1
            
        mask_actin = actin_smooth > thresh_actin
        
        # Use watershed to propagate nuclei labels into the actin intensity landscape
        # We use -actin_smooth as the topographic surface (basins at high intensity)
        # mask_actin serves as the mask of valid areas
        cell_labels = watershed(-actin_smooth, nuclei_labels, mask=mask_actin)
        
    except Exception:
        return 0.0

    # --- Feature Computation ---
    
    # Get properties for nuclei and cells
    # Note: Because we used watershed seeded by nuclei_labels, the label indices match.
    # Label '1' in nuclei_labels corresponds to Label '1' in cell_labels.
    
    props_nuc = regionprops(nuclei_labels)
    props_cell = regionprops(cell_labels, intensity_image=actin_norm)
    
    # Create a lookup for cell properties by label
    cell_props_map = {p.label: p for p in props_cell}
    
    offsets = []
    
    for nuc in props_nuc:
        label_id = nuc.label
        
        # Check if corresponding cell body exists
        if label_id in cell_props_map:
            cell = cell_props_map[label_id]
            
            # Filter small debris
            if nuc.area < 50 or cell.area < 100:
                continue
                
            # Get centroids (row, col)
            yc_n, xc_n = nuc.centroid
            yc_c, xc_c = cell.centroid
            
            # Calculate Euclidean distance
            dist = np.sqrt((xc_n - xc_c)**2 + (yc_n - yc_c)**2)
            offsets.append(dist)

    # --- Aggregation ---
    if not offsets:
        return 0.0
        
    # Return the mean offset across all valid cells
    result = np.mean(offsets)

    return float(result)
