def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import regionprops
    from skimage.morphology import binary_dilation, disk

    # 1. Data Loading and Validation
    # Convert to float32 for processing
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensions: Expected (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # 2. Channel Extraction
    # Channel 1 is Tubulin (Green) based on dataset description
    tubulin_channel = arr[..., 1]
    
    # 3. Normalization
    # Normalize tubulin channel to 0-1 range robustly
    p_min, p_max = np.percentile(tubulin_channel, (1, 99))
    if p_max > p_min:
        tubulin_norm = (tubulin_channel - p_min) / (p_max - p_min)
    else:
        tubulin_norm = tubulin_channel
    tubulin_norm = np.clip(tubulin_norm, 0.0, 1.0)

    # 4. Segmentation Handling
    # We need at least a nuclei mask to define the "perinuclear" region.
    # We ideally also want a cell mask to define the "periphery" (cell body minus perinuclear).
    
    nuclei_mask = None
    cell_mask = None

    # Check provided masks
    if len(segmentation_masks) >= 1:
        # Assuming the first mask is likely nuclei or cells. 
        # In many pipelines, if only one is provided, it might be nuclei.
        # If two are provided, often [cells, nuclei] or [nuclei, cells].
        # Let's try to infer or use standard conventions.
        # Based on common BBBC021 data structures, often we get separate files.
        # Let's assume:
        # If 2 masks: index 0 = cell/cytoplasm, index 1 = nuclei (or vice versa, need to be careful).
        # However, usually nuclei are smaller and more numerous/distinct.
        
        # Strategy: Use the masks provided.
        # If we have 2 masks, we can try to distinguish them by coverage or assume order.
        # A common convention in these challenges is often (nuclei, cells) or just (cells) with nuclei implicit.
        # Let's look at the prompt's example: "seg_mask_1: cyto, seg_mask_2: nuclei".
        # Let's try to identify which is which by size if 2 are present, or just use what we have.
        
        mask1 = segmentation_masks[0]
        if len(segmentation_masks) >= 2:
            mask2 = segmentation_masks[1]
            
            # Heuristic: Nuclei usually cover less area than whole cells
            area1 = np.sum(mask1 > 0)
            area2 = np.sum(mask2 > 0)
            
            if area1 < area2:
                nuclei_mask = mask1
                cell_mask = mask2
            else:
                nuclei_mask = mask2
                cell_mask = mask1
        else:
            # Only 1 mask. If it's nuclei, we can approximate cell body by dilation.
            # If it's cells, we can't easily find nuclei without a channel.
            # Let's assume it's nuclei because perinuclear requires a nuclear reference.
            nuclei_mask = mask1
            # Create an approximate cell mask by dilating nuclei significantly
            # or thresholding the actin/tubulin channels if needed.
            # Here, we'll try to threshold the image for a cell mask fallback.
            # Simple Otsu-like threshold on mean of Actin+Tubulin
            mean_intensity = np.mean(arr[..., 0:2], axis=2)
            threshold = np.percentile(mean_intensity, 15) # Background is usually dark
            cell_mask = (mean_intensity > threshold).astype(int)
            # Label the fallback cell mask to match nuclei labels roughly (intersection)
            # This is complex, so we'll stick to a simpler per-nucleus analysis if only 1 mask.

    # Fallback if no masks provided: Simple thresholding on DAPI (Ch2) for nuclei
    if nuclei_mask is None:
        dapi_channel = arr[..., 2]
        thresh_dapi = np.percentile(dapi_channel, 95) * 0.5
        nuclei_mask_bool = dapi_channel > thresh_dapi
        # Clean up
        nuclei_mask_bool = binary_dilation(nuclei_mask_bool, disk(2))
        nuclei_mask, _ = ndimage.label(nuclei_mask_bool)
        
    # Ensure masks are integer labels
    nuclei_mask = nuclei_mask.astype(int)
    if cell_mask is not None:
        cell_mask = cell_mask.astype(int)

    # 5. Feature Calculation: Perinuclear vs Peripheral Ratio
    # Definition:
    # Perinuclear: Region immediately surrounding the nucleus (e.g., ring of width 5-10 pixels).
    # Peripheral: The rest of the cell body (Cell Mask - Nucleus - Perinuclear).
    # If no cell mask is robust, we define Peripheral as a further ring.
    
    # Parameters based on feedback: Radius 5-10 pixels.
    perinuclear_width = 8 
    
    props = regionprops(nuclei_mask)
    
    ratios = []
    
    for prop in props:
        # Get the bounding box of the nucleus to work on a smaller slice
        minr, minc, maxr, maxc = prop.bbox
        
        # Add padding for the dilation
        pad = perinuclear_width + 5
        minr_pad = max(0, minr - pad)
        minc_pad = max(0, minc - pad)
        maxr_pad = min(arr.shape[0], maxr + pad)
        maxc_pad = min(arr.shape[1], maxc + pad)
        
        # Extract local masks and image
        local_nuclei_mask = (nuclei_mask[minr_pad:maxr_pad, minc_pad:maxc_pad] == prop.label)
        local_tubulin = tubulin_norm[minr_pad:maxr_pad, minc_pad:maxc_pad]
        
        # Define Perinuclear Region
        # Dilate nucleus by 'perinuclear_width'
        # Structure element: disk
        selem = disk(perinuclear_width)
        dilated_nucleus = binary_dilation(local_nuclei_mask, selem)
        
        # Perinuclear region = Dilated Nucleus - Original Nucleus
        perinuclear_mask = np.logical_and(dilated_nucleus, ~local_nuclei_mask)
        
        # Define Peripheral Region
        # If we have a specific cell mask, use it.
        # Otherwise, define it as a further dilation or the rest of the local area that has signal.
        peripheral_mask = None
        
        if cell_mask is not None:
            # Extract local cell mask
            # We need to find the cell label corresponding to this nucleus.
            # Usually, they overlap.
            local_cell_labels = cell_mask[minr_pad:maxr_pad, minc_pad:maxc_pad]
            # Find the most frequent cell label under the nucleus mask
            # (Assuming 1:1 mapping or containment)
            overlap = local_cell_labels[local_nuclei_mask]
            if overlap.size > 0:
                # Mode of labels
                vals, counts = np.unique(overlap, return_counts=True)
                # Ignore background 0 if possible, unless nucleus is in background (error)
                valid_indices = vals > 0
                if np.any(valid_indices):
                    cell_label = vals[valid_indices][np.argmax(counts[valid_indices])]
                    local_specific_cell_mask = (local_cell_labels == cell_label)
                    
                    # Peripheral = Cell Body - (Nucleus + Perinuclear)
                    # i.e., Cell Body - Dilated Nucleus
                    peripheral_mask = np.logical_and(local_specific_cell_mask, ~dilated_nucleus)
        
        # Fallback if no cell mask or matching failed: Use a second, wider ring
        if peripheral_mask is None or np.sum(peripheral_mask) == 0:
            # Define peripheral as a ring from width to width*3
            selem_outer = disk(perinuclear_width * 3)
            dilated_outer = binary_dilation(local_nuclei_mask, selem_outer)
            peripheral_mask = np.logical_and(dilated_outer, ~dilated_nucleus)
            
            # Mask by signal intensity to avoid measuring empty background
            # Assume background is very dark in normalized image
            signal_mask = local_tubulin > 0.05
            peripheral_mask = np.logical_and(peripheral_mask, signal_mask)

        # Calculate Intensities
        if np.sum(perinuclear_mask) > 0 and np.sum(peripheral_mask) > 0:
            mean_peri_nuc = np.mean(local_tubulin[perinuclear_mask])
            mean_peripheral = np.mean(local_tubulin[peripheral_mask])
            
            # Avoid division by zero
            if mean_peripheral > 1e-6:
                ratio = mean_peri_nuc / mean_peripheral
                ratios.append(ratio)
            elif mean_peri_nuc > 1e-6:
                # Peripheral is 0, Perinuclear is > 0 -> High ratio
                ratios.append(10.0) # Cap at a reasonable high value
            else:
                # Both zero
                ratios.append(1.0)

    # 6. Aggregation
    if not ratios:
        return 0.0
        
    # Return the mean ratio across all valid cells
    # We use median to be robust against outliers (segmentation errors)
    result = np.median(ratios)
    
    return float(result)
