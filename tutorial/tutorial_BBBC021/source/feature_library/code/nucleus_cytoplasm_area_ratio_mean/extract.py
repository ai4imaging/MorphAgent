def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)
    
    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3)
    if arr.ndim != 3:
        return 0.0
    
    # This feature relies heavily on segmentation masks.
    # We need at least two masks to distinguish nucleus from cytoplasm/cell body.
    # Typically, mask 0 is the nucleus and mask 1 is the whole cell (or vice versa).
    # Based on standard biological pipelines (e.g., CellProfiler), usually:
    # - Primary objects = Nuclei
    # - Secondary objects = Cells (expanded from nuclei)
    
    # Check if we have enough masks
    if len(segmentation_masks) < 2:
        # Fallback: If we don't have masks, we can't reliably compute N/C ratio 
        # without implementing a full segmentation pipeline here, which is error-prone 
        # and computationally expensive.
        # However, we can try a very crude threshold-based estimation using the channels.
        # Channel 2 = Nucleus (DAPI), Channel 0/1 = Cytoplasm (Actin/Tubulin).
        
        # Extract channels
        cyto_ch = np.maximum(arr[..., 0], arr[..., 1]) # Max of Actin/Tubulin
        nuc_ch = arr[..., 2] # DAPI
        
        # Normalize
        def normalize(c):
            vmax = np.percentile(c, 99) if c.size > 0 else 1.0
            return np.clip(c / (vmax + 1e-6), 0, 1)
            
        nuc_norm = normalize(nuc_ch)
        cyto_norm = normalize(cyto_ch)
        
        # Simple thresholding (Otsu-like approximation or fixed)
        # Using fixed conservative thresholds for robustness
        nuc_mask_binary = nuc_norm > 0.3
        # Cell body usually includes the nucleus area + cytoplasm
        cell_mask_binary = (cyto_norm > 0.15) | nuc_mask_binary
        
        # Label connected components to treat as instances
        nuc_labels = label(nuc_mask_binary)
        cell_labels = label(cell_mask_binary)
        
        # If we still don't have valid labeled masks, return 0
        if nuc_labels.max() == 0:
            return 0.0
            
        # Assign generated masks for processing below
        nucleus_mask = nuc_labels
        cell_mask = cell_labels
        
        # Note: Without matched instance IDs (Nucleus 1 -> Cell 1), 
        # we can only compute the global area ratio, not per-cell mean ratio.
        # Global Ratio = Total Nuclear Area / Total Cytoplasmic Area
        total_nuc_area = np.sum(nuc_mask_binary)
        total_cell_area = np.sum(cell_mask_binary)
        total_cyto_area = total_cell_area - total_nuc_area
        
        if total_cyto_area <= 0:
            return 0.0
            
        return float(total_nuc_area / total_cyto_area)

    else:
        # We have segmentation masks provided by the system
        # Assuming:
        # segmentation_masks[0] -> Nuclei (based on typical ordering)
        # segmentation_masks[1] -> Cells (whole cell body)
        # We need to verify which is which based on size. Nuclei are smaller.
        
        mask_a = segmentation_masks[0]
        mask_b = segmentation_masks[1]
        
        # Count non-zero pixels to guess which is nucleus (smaller) and which is cell (larger)
        area_a = np.count_nonzero(mask_a)
        area_b = np.count_nonzero(mask_b)
        
        if area_a < area_b:
            nucleus_mask = mask_a
            cell_mask = mask_b
        else:
            nucleus_mask = mask_b
            cell_mask = mask_a

    # Ensure masks are labeled (instance segmentation)
    # If they are binary (max value 1), label them.
    if nucleus_mask.max() <= 1:
        nucleus_mask = label(nucleus_mask > 0)
    if cell_mask.max() <= 1:
        cell_mask = label(cell_mask > 0)

    # Fast computation using bincount
    # This assumes that the labels are matched (Label 1 in nucleus_mask corresponds to Label 1 in cell_mask).
    # This is standard for "Primary" -> "Secondary" object pipelines.
    
    # Get max label ID to set array size
    max_id = max(nucleus_mask.max(), cell_mask.max())
    
    if max_id == 0:
        return 0.0

    # Calculate areas for each label ID
    # minlength ensures both arrays are the same size
    n_areas = np.bincount(nucleus_mask.ravel(), minlength=max_id + 1)
    c_areas = np.bincount(cell_mask.ravel(), minlength=max_id + 1)
    
    # Ignore background (index 0)
    n_areas = n_areas[1:]
    c_areas = c_areas[1:]
    
    # Filter for valid cells: must exist in both masks
    valid_indices = (n_areas > 0) & (c_areas > 0)
    
    if not np.any(valid_indices):
        return 0.0
        
    valid_n_areas = n_areas[valid_indices].astype(np.float64)
    valid_c_areas = c_areas[valid_indices].astype(np.float64)
    
    # Calculate Cytoplasm Area
    # Ideally: Cell Area - Nucleus Area
    # However, sometimes masks are disjoint or strictly cytoplasmic.
    # If Cell Area is significantly larger than Nucleus Area, we assume Cell Mask includes Nucleus.
    # If Cell Area is roughly equal to or smaller than Nucleus Area (impossible if inclusive), 
    # it might be a "Ring" mask (cytoplasm only).
    
    # We calculate the cytoplasm area assuming Cell Mask is the WHOLE cell.
    cyto_areas = valid_c_areas - valid_n_areas
    
    # Handle edge cases where segmentation might be imperfect (Nucleus > Cell)
    # or where the provided mask was already just the cytoplasm.
    # If cyto_area is very small or negative, check if valid_c_areas represents just cytoplasm.
    # Heuristic: If >50% of cells have negative derived cyto area, assume the second mask was ALREADY cytoplasm.
    negative_cyto_count = np.sum(cyto_areas <= 0)
    if negative_cyto_count > (len(cyto_areas) * 0.5):
        # Assumption switch: The second mask was already the cytoplasm
        cyto_areas = valid_c_areas
    else:
        # Clip negative values to a small epsilon to avoid division errors
        cyto_areas = np.maximum(cyto_areas, 1.0)

    # Calculate Ratio: Nucleus Area / Cytoplasm Area
    ratios = valid_n_areas / cyto_areas
    
    # Filter out infinite or NaN ratios
    ratios = ratios[np.isfinite(ratios)]
    
    if len(ratios) == 0:
        return 0.0
        
    # Return the mean ratio
    result = np.mean(ratios)
    
    return float(result)
