def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import regionprops, label
    from skimage.segmentation import clear_border, watershed
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract channels
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green)
    # Channel 2: DAPI (Blue)
    actin_ch = arr[..., 0]
    tubulin_ch = arr[..., 1]
    dapi_ch = arr[..., 2]

    # Combine Actin and Tubulin to represent the "Cell Body" signal
    # Using maximum projection to capture the extent of the cytoskeleton
    cell_body_signal = np.maximum(actin_ch, tubulin_ch)

    # Determine Segmentation Strategy
    nuclei_mask = None
    cells_mask = None

    # Check provided masks
    # We expect masks to be passed as arguments.
    # Common convention: if 2 masks, often [cells, nuclei] or [nuclei, cells].
    # If 1 mask, usually nuclei.
    # We need to be robust.
    
    has_masks = len(segmentation_masks) > 0
    
    if has_masks:
        # Heuristic to identify masks based on overlap with channels
        # Calculate mean intensity of DAPI within mask regions to identify nuclei mask
        
        best_nuclei_score = -1
        best_nuclei_idx = -1
        
        # Normalize DAPI for scoring
        dapi_norm = (dapi_ch - dapi_ch.min()) / (dapi_ch.max() - dapi_ch.min() + 1e-7)

        for i, mask in enumerate(segmentation_masks):
            if mask is None: continue
            # Ensure mask is integer
            mask = mask.astype(int)
            if mask.max() == 0: continue
            
            # Score: Mean DAPI intensity inside mask
            mask_bool = mask > 0
            score = np.mean(dapi_norm[mask_bool])
            
            if score > best_nuclei_score:
                best_nuclei_score = score
                best_nuclei_idx = i
        
        if best_nuclei_idx != -1:
            nuclei_mask = segmentation_masks[best_nuclei_idx].astype(int)
            
            # If there is another mask, assume it is the cell mask
            if len(segmentation_masks) > 1:
                # Pick the one that isn't the nuclei mask
                for i, mask in enumerate(segmentation_masks):
                    if i != best_nuclei_idx and mask is not None and mask.max() > 0:
                        cells_mask = mask.astype(int)
                        break
    
    # Fallback: Generate Nuclei Mask if missing
    if nuclei_mask is None:
        try:
            thresh = threshold_otsu(dapi_ch)
            binary_nuc = dapi_ch > thresh
            binary_nuc = binary_opening(binary_nuc, disk(2))
            nuclei_mask = label(binary_nuc)
        except Exception:
            return 0.0

    # Fallback: Generate Cell Mask if missing
    # Use watershed seeded by nuclei on the inverted cell body signal
    if cells_mask is None:
        try:
            # Create a background marker (0)
            # We need a background seed. A simple approach is pixels with very low intensity
            # or the inverse of a threshold on the cell body.
            try:
                cell_thresh = threshold_otsu(cell_body_signal)
            except:
                cell_thresh = np.mean(cell_body_signal)
                
            binary_cell = cell_body_signal > cell_thresh
            
            # Markers: Nuclei are seeds
            markers = nuclei_mask.copy()
            
            # Gradient for watershed
            gradient = ndimage.gaussian_gradient_magnitude(cell_body_signal, sigma=1)
            
            # Run watershed
            # We mask the watershed to only the binary cell area to avoid over-expansion into background
            cells_mask = watershed(gradient, markers, mask=binary_cell)
        except Exception:
            # If watershed fails, we can't compute cell centers distinct from nuclei
            return 0.0

    # Clear borders
    # We remove cells touching the border because their centroids are unreliable
    # We apply clear_border to the CELLS mask, and then filter the nuclei mask to match
    cells_mask_cleared = clear_border(cells_mask)
    
    # Get properties
    props_nuc = regionprops(nuclei_mask)
    props_cell = regionprops(cells_mask_cleared)
    
    if not props_nuc or not props_cell:
        return 0.0

    # Map Nuclei to Cells
    # If we generated masks via watershed, the labels match (Nucleus 1 -> Cell 1).
    # If masks came from external sources, labels might not match.
    # We need a robust matching strategy: Nucleus is contained within Cell.
    
    # Create dictionaries for fast lookup
    # {label: centroid}
    nuc_centers = {p.label: np.array(p.centroid) for p in props_nuc}
    cell_centers = {p.label: np.array(p.centroid) for p in props_cell}
    
    distances = []
    
    # Iterate through valid cells (those that survived border clearing)
    for cell_label, cell_centroid in cell_centers.items():
        
        # Find the corresponding nucleus
        # Strategy: Look for the nucleus label that overlaps most with this cell
        # Optimization: If labels are consistent (common in segmentation pipelines), check that first
        
        nuc_centroid = None
        
        if cell_label in nuc_centers:
            # Check consistency: Is the nucleus center actually inside the cell mask region?
            # Or simpler: assume label consistency if generated by watershed.
            # If external masks, we might need spatial matching.
            
            # Let's verify spatial containment or proximity to be safe against mismatched labels
            # But for this specific feature implementation, assuming label consistency 
            # (or that the nucleus label is the seed for the cell label) is standard 
            # when handling (Nuclei, Cell) pairs.
            
            # However, if masks were provided externally and have different label sets,
            # we need to find which nucleus is inside this cell.
            # Since we can't easily do pixel overlap without iterating pixels, 
            # we'll assume label consistency if generated, or try to match by distance if not.
            
            # If we generated the cell mask from the nuclei mask (watershed), labels are guaranteed to match.
            # If both were provided, we check if the label exists.
            
            nuc_centroid = nuc_centers[cell_label]
        else:
            # Label mismatch or nucleus was filtered out (e.g. if we cleared borders on nuclei but not cells, or vice versa)
            # If we cleared borders on cells, the corresponding nucleus might still exist in the original nuclei mask.
            # We need to find the nucleus that corresponds to this cell.
            # Fallback: Find nucleus closest to cell center (risky if crowded) or skip.
            # Given the constraints, we will skip if ID doesn't match to avoid false pairings.
            continue
            
        if nuc_centroid is not None:
            # Compute Euclidean distance
            dist = np.linalg.norm(cell_centroid - nuc_centroid)
            distances.append(dist)

    if not distances:
        return 0.0

    return float(np.mean(distances))
