def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import opening, disk
    from skimage.segmentation import watershed

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not 3 channels, we can't reliably distinguish nucleus vs cell body for this specific feature
        # unless masks are provided. If masks are provided, we might proceed, but let's be safe.
        if len(segmentation_masks) < 2:
            return 0.0

    # Define channels based on dataset description
    # Ch0: Actin (Red), Ch1: Tubulin (Green), Ch2: DAPI (Blue)
    # Nucleus signal is primarily in Ch2
    # Cell body signal is primarily in Ch0 and Ch1 (and Ch2)
    
    nuclei_mask = None
    cell_mask = None

    # --- Step 1: Obtain Segmentation Masks ---
    
    # Check if valid segmentation masks are provided
    # We need two distinct masks: one for nuclei, one for cells (or cytoplasm)
    # If only one mask is provided, we assume it's a cell mask and we still need to segment nuclei from the image.
    # However, standard practice in this context usually provides either (nuclei, cells) or nothing.
    
    if len(segmentation_masks) >= 2:
        # Assuming order: segmentation_masks[0] is often nuclei or cells. 
        # Without explicit metadata on which is which, we can try to infer or assume a standard order.
        # A common convention is often Nuclei first, then Cells, or vice versa.
        # Let's try to infer based on size/containment if possible, or fallback to generation if ambiguous.
        # Given the prompt doesn't specify mask order, we will implement a robust fallback:
        # We will GENERATE our own masks from the raw image to ensure we know exactly which is nucleus and which is cell.
        # This is safer than guessing which mask is which in the *args.
        # However, if the prompt implies using provided masks, we should try.
        # Let's stick to the safest route: Use the raw channels to generate fresh masks. 
        # This ensures the "Nucleus" mask actually corresponds to the DAPI channel.
        pass 

    # Implementation Strategy: 
    # Regardless of input masks (which might be ambiguous in order), we can robustly generate them 
    # from the 3-channel image because the channels are biologically specific.
    
    # 1.1 Generate Nuclei Mask (from Channel 2 - DAPI)
    dapi_ch = arr[..., 2]
    # Normalize DAPI
    dapi_max = np.percentile(dapi_ch, 99.9)
    if dapi_max > 0:
        dapi_ch = dapi_ch / dapi_max
    
    # Threshold and Label Nuclei
    try:
        thresh_nuc = threshold_otsu(dapi_ch)
        nuclei_bool = dapi_ch > thresh_nuc
        # Clean up noise
        nuclei_bool = opening(nuclei_bool, disk(2))
        nuclei_labels = label(nuclei_bool)
    except Exception:
        return 0.0

    # 1.2 Generate Cell Mask (from combined channels)
    # Combine Actin (0) and Tubulin (1) for cell body, plus DAPI (2) to ensure coverage
    cell_signal = np.mean(arr, axis=2) # Average of all channels is a decent proxy for cell body
    
    # Normalize Cell Signal
    cell_max = np.percentile(cell_signal, 99.9)
    if cell_max > 0:
        cell_signal = cell_signal / cell_max
        
    try:
        thresh_cell = threshold_otsu(cell_signal)
        cell_bool = cell_signal > thresh_cell
        
        # Watershed to separate touching cells based on nuclei markers
        # This ensures every cell region corresponds to exactly one nucleus (ideal case)
        distance = ndimage.distance_transform_edt(cell_bool)
        # We use the nuclei_labels as markers. 
        # Cells without nuclei will be background. Nuclei without cell body signal will be expanded.
        cell_labels = watershed(-distance, nuclei_labels, mask=cell_bool)
    except Exception:
        return 0.0

    # --- Step 2: Compute Centroids and Distances ---
    
    # We now have:
    # nuclei_labels: labeled image of nuclei
    # cell_labels: labeled image of cells, where the label ID matches the nucleus ID (due to watershed)
    
    props_nuc = regionprops(nuclei_labels)
    props_cell = regionprops(cell_labels)
    
    # Create a dictionary for fast lookup of cell properties by label
    cell_props_map = {p.label: p for p in props_cell}
    
    distances = []
    
    for nuc in props_nuc:
        label_id = nuc.label
        
        # Check if a corresponding cell exists
        if label_id in cell_props_map:
            cell = cell_props_map[label_id]
            
            # Get centroids (row, col)
            yc_n, xc_n = nuc.centroid
            yc_c, xc_c = cell.centroid
            
            # Calculate Euclidean distance
            dist = np.sqrt((yc_n - yc_c)**2 + (xc_n - xc_c)**2)
            distances.append(dist)
            
    # --- Step 3: Aggregate ---
    
    if not distances:
        return 0.0
        
    result = np.mean(distances)

    return float(result)
