def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.segmentation import watershed
    from skimage.morphology import opening, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Channel mapping based on dataset description:
    # Ch0: Actin (Red) -> Cytoskeleton
    # Ch1: Tubulin (Green) -> Cytoskeleton
    # Ch2: DAPI (Blue) -> Nucleus
    ch_actin = arr[..., 0]
    ch_tubulin = arr[..., 1]
    ch_dapi = arr[..., 2]

    # Normalize channels to [0, 1] for processing
    def normalize(c):
        mx = c.max()
        return c / mx if mx > 0 else c
    
    ch_actin = normalize(ch_actin)
    ch_tubulin = normalize(ch_tubulin)
    ch_dapi = normalize(ch_dapi)

    # --- Segmentation Logic ---
    # We need two labeled masks: nuclei_labels and cell_labels (whole cell)
    
    nuclei_labels = None
    cell_labels = None

    # Check if valid segmentation masks are provided
    # We expect masks to be passed in a specific order if they exist, but we must be robust.
    # If masks are provided, we assume:
    # - If 2 masks: 1st is likely cells/cytoplasm, 2nd is nuclei (or vice versa, need to check overlap)
    # - If 1 mask: likely whole cells, hard to get N/C ratio without nuclei specific mask.
    
    # Given the variability of external masks, and the specific requirement for N/C ratio 
    # (which requires strict correspondence between a nucleus and its cytoplasm),
    # it is often safer to perform a consistent segmentation pipeline internally if the 
    # provided masks are not explicitly labeled as "nuclei" and "cells".
    # However, if masks are present, we try to use them.
    
    if len(segmentation_masks) >= 2:
        # Heuristic: Nuclei mask usually has smaller total area than cell mask
        m1 = segmentation_masks[0]
        m2 = segmentation_masks[1]
        
        # Ensure they are labeled arrays
        if m1.ndim == 2 and m2.ndim == 2:
            area1 = np.count_nonzero(m1)
            area2 = np.count_nonzero(m2)
            
            if area1 < area2:
                nuclei_labels = m1
                cell_labels = m2
            else:
                nuclei_labels = m2
                cell_labels = m1
    
    # Fallback: Internal Segmentation Pipeline
    if nuclei_labels is None or cell_labels is None:
        # 1. Segment Nuclei (DAPI - Ch2)
        # Smooth slightly
        dapi_smooth = ndimage.gaussian_filter(ch_dapi, sigma=2)
        try:
            thresh_nuc = threshold_otsu(dapi_smooth)
        except ValueError: # Handle empty images
            thresh_nuc = 0.1
            
        mask_nuc = dapi_smooth > thresh_nuc
        # Clean up noise
        mask_nuc = opening(mask_nuc, disk(2))
        nuclei_labels = label(mask_nuc)

        # 2. Segment Whole Cells (Actin Ch0 + Tubulin Ch1)
        # Combine cytoskeletal channels for better cell body definition
        cyto_signal = np.maximum(ch_actin, ch_tubulin)
        cyto_smooth = ndimage.gaussian_filter(cyto_signal, sigma=2)
        
        try:
            thresh_cell = threshold_otsu(cyto_smooth)
        except ValueError:
            thresh_cell = 0.1
            
        # The cell mask should include the nuclei
        mask_cell_fg = (cyto_smooth > thresh_cell) | mask_nuc
        
        # 3. Instance Segmentation via Watershed
        # Use nuclei as markers to split the cytoplasm mask into individual cells
        # We calculate the distance transform for better watershed boundaries if needed,
        # but standard watershed on the intensity or just mask geometry works for simple separation.
        # Here we use the inverted intensity as the "elevation" map for watershed
        # so boundaries form at low intensity regions.
        elevation_map = -cyto_smooth
        
        # Run watershed
        # mask=mask_cell_fg ensures we only label the foreground
        cell_labels = watershed(elevation_map, nuclei_labels, mask=mask_cell_fg)

    # --- Feature Computation: Mean N/C Ratio ---
    
    # Get properties for nuclei and cells
    # We need to match them. Since we used watershed seeded by nuclei, 
    # the label ID 'k' in cell_labels corresponds to the nucleus with label ID 'k' in nuclei_labels.
    
    props_nuclei = regionprops(nuclei_labels)
    props_cells = regionprops(cell_labels)
    
    # Create a dictionary for fast lookup of cell areas by label
    cell_areas = {p.label: p.area for p in props_cells}
    
    ratios = []
    
    for n_prop in props_nuclei:
        label_id = n_prop.label
        
        # Check if this nucleus has a corresponding cell body
        if label_id in cell_areas:
            area_nucleus = n_prop.area
            area_whole_cell = cell_areas[label_id]
            
            # Cytoplasm area = Whole Cell - Nucleus
            # Note: In watershed, the cell label includes the nucleus pixels.
            area_cytoplasm = area_whole_cell - area_nucleus
            
            # Filter out artifacts
            if area_cytoplasm > 0 and area_nucleus > 10: # Minimum size check
                ratio = area_nucleus / float(area_cytoplasm)
                ratios.append(ratio)
            elif area_cytoplasm == 0 and area_nucleus > 10:
                # If nucleus fills the whole cell (segmentation error or very high ratio), 
                # we can cap it or ignore. Ignoring is safer to avoid skewing mean with outliers.
                pass

    if not ratios:
        return 0.0

    result = np.mean(ratios)
    return float(result)
