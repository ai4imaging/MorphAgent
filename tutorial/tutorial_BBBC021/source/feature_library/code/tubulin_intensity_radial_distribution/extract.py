def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_closing, disk, binary_dilation

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality and validate input
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract relevant channels
    # Channel 1: Tubulin (Green) - The signal to measure
    # Channel 2: DAPI (Blue) - The nuclear reference
    # Channel 0: Actin (Red) - Used for cell body estimation if no mask provided
    tubulin_img = arr[:, :, 1]
    dapi_img = arr[:, :, 2]
    actin_img = arr[:, :, 0]

    # Normalize intensity to [0, 1] for processing
    # (Though ratio calculation is scale-invariant, this helps with thresholding)
    def normalize(x):
        vmax = np.percentile(x, 99.5) if x.size > 0 else 1.0
        if vmax <= 0: return x
        return np.clip(x / vmax, 0, 1)

    tubulin_norm = normalize(tubulin_img)
    dapi_norm = normalize(dapi_img)
    actin_norm = normalize(actin_img)

    # --- Segmentation Logic ---
    # We need two masks: 'nuclei_mask' and 'cell_mask'
    
    nuclei_mask = None
    cell_mask = None

    # Check if valid segmentation masks are provided
    # We look for masks that match the image spatial dimensions (512, 512)
    valid_masks = [m for m in segmentation_masks if m is not None and m.shape[:2] == arr.shape[:2]]

    if len(valid_masks) >= 2:
        # Heuristic: usually nuclei are smaller/more numerous or labeled specifically. 
        # Without metadata, we assume the order or size. 
        # Often datasets provide [cell_mask, nuclei_mask] or vice versa.
        # Let's try to distinguish by containment. Nuclei should be inside cells.
        m1 = valid_masks[0] > 0
        m2 = valid_masks[1] > 0
        
        # If m1 is contained in m2, m1 is likely nuclei
        intersection = np.logical_and(m1, m2)
        if np.sum(m1) < np.sum(m2) and (np.sum(intersection) / (np.sum(m1) + 1e-6) > 0.9):
            nuclei_mask = m1
            cell_mask = m2
        else:
            nuclei_mask = m2
            cell_mask = m1
            
    elif len(valid_masks) == 1:
        # If only one mask, assume it's the cell mask if it's large, or nuclei if small.
        # But for this specific feature (radial distribution), we critically need the nucleus.
        # We will fallback to computing the missing one.
        provided_mask = valid_masks[0] > 0
        
        # Generate ad-hoc nuclei mask to check against provided mask
        try:
            thresh_nuc = threshold_otsu(dapi_norm)
            adhoc_nuclei = dapi_norm > thresh_nuc
        except:
            adhoc_nuclei = dapi_norm > 0.2
            
        # If provided mask overlaps significantly with adhoc nuclei but is much larger, it's cell.
        if np.sum(provided_mask) > 2 * np.sum(adhoc_nuclei):
            cell_mask = provided_mask
            nuclei_mask = adhoc_nuclei
        else:
            nuclei_mask = provided_mask
            # Create ad-hoc cell mask
            try:
                # Combine actin and tubulin for cell body
                composite = np.maximum(actin_norm, tubulin_norm)
                thresh_cell = threshold_otsu(composite)
                cell_mask = composite > thresh_cell
            except:
                cell_mask = composite > 0.1
    else:
        # No masks provided: Compute both
        try:
            thresh_nuc = threshold_otsu(dapi_norm)
            nuclei_mask = binary_closing(dapi_norm > thresh_nuc, disk(2))
        except:
            nuclei_mask = dapi_norm > 0.1 # Fallback

        try:
            composite = np.maximum(actin_norm, tubulin_norm)
            thresh_cell = threshold_otsu(composite)
            cell_mask = binary_closing(composite > thresh_cell, disk(3))
        except:
            cell_mask = composite > 0.1 # Fallback

    # Ensure masks are boolean
    nuclei_mask = nuclei_mask.astype(bool)
    cell_mask = cell_mask.astype(bool)

    # Ensure nuclei are inside cells (clean up segmentation artifacts)
    cell_mask = np.logical_or(cell_mask, nuclei_mask)
    
    # Label the cells to process them individually
    # We use the nuclei as seeds to identify individual cells
    labeled_nuclei, num_nuclei = label(nuclei_mask, return_num=True)
    
    if num_nuclei == 0:
        return 0.0

    # --- Feature Computation: Radial Distribution ---
    
    # We need to define "Inner 50%" vs "Outer 50%" of the cytoplasm.
    # Method: Normalized Distance Transform per cell.
    # R = Dist(pixel, nucleus) / (Dist(pixel, nucleus) + Dist(pixel, background))
    # R goes from 0 (at nuclear boundary) to 1 (at cell boundary).
    
    # Optimization: Instead of iterating per cell (slow in Python), we can compute global distance maps
    # masked by specific cell regions if cells are not touching. 
    # However, cells often touch. A watershed separation is ideal, but for this scalar feature,
    # we can approximate by iterating over regionprops which is safer.

    ratios = []

    # Get properties of labeled nuclei
    nuc_props = regionprops(labeled_nuclei)
    
    # To associate cytoplasm with nuclei, we can use a watershed or simply mask the global calculation
    # if the image is sparse. Given MCF-7 can be confluent, we'll do a simplified per-object approach
    # by using the bounding box of the nucleus expanded to find the cell.
    
    # Better approach for robustness:
    # 1. Compute Distance from Nuclei (inverted: 0 at nuclei, increasing outwards)
    # 2. Compute Distance from Background (0 at background, increasing inwards)
    # This requires a labeled cell mask matching the nuclei.
    
    # Let's generate a labeled cell mask seeded by nuclei
    # Distance transform for watershed
    distance = ndimage.distance_transform_edt(cell_mask)
    # Watershed is not available in the allowed imports list (scipy.ndimage doesn't have watershed, skimage.segmentation does but wasn't explicitly allowed in "Basic numeric").
    # However, "Optional morphology" allows skimage.measure.label.
    # We will use a simpler approach: Assign cytoplasm pixels to nearest nucleus.
    
    # 1. Distance transform from all nuclei
    dist_from_nuclei, nearest_nucleus_indices = ndimage.distance_transform_edt(
        ~nuclei_mask, return_indices=True
    )
    
    # 2. Create a labeled cell map where each pixel in 'cell_mask' takes the ID of the nearest nucleus
    # 'nearest_nucleus_indices' gives coordinates. We map these to the label at that coordinate.
    # This is effectively a Voronoi tessellation inside the cell mask.
    labeled_cells = np.zeros_like(labeled_nuclei)
    
    # Only process pixels that are part of a cell
    valid_pixels = cell_mask
    
    if not np.any(valid_pixels):
        return 0.0

    # Map indices to labels
    # nearest_nucleus_indices is (2, H, W). 
    # We want labeled_nuclei[nearest_nucleus_indices[0], nearest_nucleus_indices[1]]
    # But we only care about pixels inside 'cell_mask'
    
    # Optimization: We can just iterate over the unique labels found in labeled_nuclei
    # But constructing the full labeled_cells map via Voronoi is efficient enough.
    
    # Flatten for indexing
    flat_labels = labeled_nuclei[nearest_nucleus_indices[0, valid_pixels], nearest_nucleus_indices[1, valid_pixels]]
    labeled_cells[valid_pixels] = flat_labels
    
    # Now we have labeled_cells matching labeled_nuclei.
    
    # Compute Normalized Radial Distance
    # Dist_Nuc: Distance from the specific nucleus boundary.
    # Since we did a global Voronoi, the 'dist_from_nuclei' computed earlier is exactly
    # the distance to the nearest nucleus. For a pixel belonging to Cell A, the nearest nucleus IS Nucleus A.
    dist_nuc = dist_from_nuclei
    
    # Dist_Bound: Distance to the background (inverse of cell mask)
    dist_bound = ndimage.distance_transform_edt(cell_mask)
    
    # Avoid division by zero
    denominator = dist_nuc + dist_bound
    denominator[denominator == 0] = 1.0
    
    normalized_radius = dist_nuc / denominator
    
    # We only care about the Cytoplasm (Cell mask - Nuclei mask)
    cytoplasm_mask = np.logical_and(cell_mask, ~nuclei_mask)
    
    # Define Zones
    # Inner: 0.0 <= R < 0.5
    # Outer: 0.5 <= R <= 1.0
    inner_mask = np.logical_and(cytoplasm_mask, normalized_radius < 0.5)
    outer_mask = np.logical_and(cytoplasm_mask, normalized_radius >= 0.5)
    
    # We want to compute the ratio per cell, then average (or median).
    # Aggregating all pixels globally might be biased by large cells.
    # Let's compute per cell.
    
    cell_props = regionprops(labeled_cells, intensity_image=tubulin_img)
    
    cell_ratios = []
    
    for prop in cell_props:
        # Get the bounding box slice
        sl = prop.slice
        
        # Extract masks and image for this cell within its bounding box
        local_label_mask = (labeled_cells[sl] == prop.label)
        local_inner = inner_mask[sl] & local_label_mask
        local_outer = outer_mask[sl] & local_label_mask
        local_tubulin = tubulin_img[sl]
        
        # Compute Mean Intensities
        # We use mean because the area of outer ring is usually larger than inner ring
        
        if np.sum(local_inner) == 0 or np.sum(local_outer) == 0:
            continue
            
        mean_inner = np.mean(local_tubulin[local_inner])
        mean_outer = np.mean(local_tubulin[local_outer])
        
        # Ratio: Perinuclear / Peripheral
        # Add epsilon to avoid div by zero
        ratio = mean_inner / (mean_outer + 1e-7)
        cell_ratios.append(ratio)

    if not cell_ratios:
        return 0.0
        
    # Return the median ratio across the population to be robust to outliers
    result = np.median(cell_ratios)

    return float(result)
