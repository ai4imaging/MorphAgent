def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Actin Channel (Channel 0)
    actin_channel = arr[:, :, 0]

    # Normalize Actin Channel
    # Robust normalization to handle potential outliers
    p99 = np.percentile(actin_channel, 99.5)
    if p99 > 0:
        actin_norm = actin_channel / p99
    else:
        actin_norm = actin_channel
    actin_norm = np.clip(actin_norm, 0.0, 1.0)

    # Segmentation Logic
    # If segmentation masks are provided, use them. Otherwise, generate one.
    # We prefer a cell mask (often derived from actin or a combination) over a nuclei mask for cell shape.
    
    labeled_cells = None
    
    if len(segmentation_masks) > 0:
        # Check available masks. Usually, masks might be [nuclei, cells] or just [nuclei].
        # We need the cell boundary mask.
        # Heuristic: if multiple masks, the one with larger average area is likely the cell mask.
        # If only one mask, we check if it covers a significant portion of the image (cytoplasm) vs small dots (nuclei).
        
        best_mask = None
        max_mean_area = 0
        
        for mask in segmentation_masks:
            if mask is None: continue
            # Ensure mask is 2D
            if mask.ndim == 3:
                mask = np.max(mask, axis=2) # Project if 3D
            
            # Quick check on labels
            # If the mask is not labeled (binary), label it
            if mask.max() == 1 and mask.dtype != bool: # Binary 0/1
                 temp_labels = label(mask)
            else:
                 temp_labels = mask.astype(int)
            
            props = regionprops(temp_labels)
            if not props: continue
            
            mean_area = np.mean([p.area for p in props])
            if mean_area > max_mean_area:
                max_mean_area = mean_area
                best_mask = temp_labels
        
        if best_mask is not None:
            labeled_cells = best_mask

    # Fallback: Generate segmentation from Actin channel if no suitable mask provided
    if labeled_cells is None:
        # Smooth slightly to reduce noise
        smooth_actin = ndimage.gaussian_filter(actin_norm, sigma=2)
        
        # Thresholding
        try:
            thresh = threshold_otsu(smooth_actin)
        except:
            thresh = 0.1
            
        # Create binary mask
        binary_mask = smooth_actin > thresh
        
        # Morphological opening to separate weakly connected cells and remove noise
        binary_mask = binary_opening(binary_mask, disk(2))
        
        # Label connected components
        labeled_cells = label(binary_mask)

    # Feature Extraction: Star-like Morphology
    # "Star-like" or "Spiked" cells typically have:
    # 1. Low Solidity: The convex hull area is much larger than the actual area due to protrusions.
    # 2. Low Form Factor (4*pi*Area / Perimeter^2): They are not circular.
    # 3. High Eccentricity is possible but not strictly required (a star can be roughly symmetric).
    # 4. Roughness/Spikiness.
    
    # Based on feedback, previous thresholds were too permissive.
    # Tightened Criteria:
    # Solidity < 0.75 (Significant indentations/spikes)
    # Form Factor < 0.50 (Far from circular)
    # Area > 100 (Ignore small debris)
    
    props = regionprops(labeled_cells)
    star_cell_count = 0.0
    
    for prop in props:
        # Filter small debris
        if prop.area < 100:
            continue
            
        # Calculate shape descriptors
        solidity = prop.solidity
        
        # Form factor (circularity)
        perimeter = prop.perimeter
        if perimeter == 0:
            continue
        form_factor = (4 * np.pi * prop.area) / (perimeter ** 2)
        
        # Eccentricity
        eccentricity = prop.eccentricity
        
        # Criteria for Star-like/Spiked Actin
        # Star cells have protrusions that significantly reduce solidity and form factor.
        # They are distinct from simple elongated cells (which have high eccentricity but high solidity).
        # They are distinct from round cells (high solidity, high form factor).
        
        # Strict thresholds based on feedback
        is_low_solidity = solidity < 0.75
        is_low_form_factor = form_factor < 0.50
        
        # To distinguish from just messy segmentation or artifacts, we might check if the cell is not extremely elongated 
        # (which might be a fiber or artifact), although some star cells can be elongated.
        # However, star shapes are often somewhat "spread out" rather than linear.
        # We ensure it's not a single thin line (eccentricity close to 1.0 with very low solidity might be a line).
        # But spikes generally lower solidity more than simple elongation does.
        
        if is_low_solidity and is_low_form_factor:
            star_cell_count += 1.0

    return float(star_cell_count)
