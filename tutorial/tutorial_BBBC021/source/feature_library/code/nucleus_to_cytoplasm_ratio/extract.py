def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu, threshold_li
    from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects
    from skimage.measure import label, regionprops
    from skimage.segmentation import watershed

    # Convert to float32 for processing
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Channel mapping based on dataset description:
    # Ch 0: Actin (Red) -> Cytoskeleton
    # Ch 1: Tubulin (Green) -> Cytoskeleton
    # Ch 2: DAPI (Blue) -> Nucleus
    actin = arr[..., 0]
    tubulin = arr[..., 1]
    dapi = arr[..., 2]

    # Normalize channels individually to handle intensity variations
    def normalize(ch):
        p99 = np.percentile(ch, 99.9)
        if p99 > 0:
            return np.clip(ch / p99, 0, 1)
        return ch

    actin_norm = normalize(actin)
    tubulin_norm = normalize(tubulin)
    dapi_norm = normalize(dapi)

    # --- Nucleus Segmentation ---
    # Use Gaussian blur to smooth noise before thresholding
    dapi_smooth = ndimage.gaussian_filter(dapi_norm, sigma=2)
    
    # Use Otsu's method for nuclei, it's generally robust for DAPI
    try:
        thresh_nuc = threshold_otsu(dapi_smooth)
    except:
        thresh_nuc = 0.1
        
    nuclei_mask = dapi_smooth > thresh_nuc
    
    # Clean up nuclei mask
    nuclei_mask = binary_opening(nuclei_mask, disk(2))
    nuclei_mask = remove_small_objects(nuclei_mask, min_size=50)
    
    # Label nuclei to separate touching cells (needed for watershed seeds)
    nuclei_labels = label(nuclei_mask)

    # --- Cell Body (Cytoplasm + Nucleus) Segmentation ---
    # Combine cytoskeletal channels. Max projection often preserves structure better than mean.
    cyto_signal = np.maximum(actin_norm, tubulin_norm)
    
    # Smooth the cytoplasmic signal significantly to connect gaps in cytoskeleton
    cyto_smooth = ndimage.gaussian_filter(cyto_signal, sigma=3)
    
    # Use Li thresholding for cytoplasm as it handles long tails (background) better than Otsu
    # which might cut off faint edges of the cell.
    try:
        thresh_cell = threshold_li(cyto_smooth)
    except:
        thresh_cell = 0.05
        
    # Create initial cell mask
    cell_mask_raw = cyto_smooth > thresh_cell
    
    # Ensure the cell mask includes the nuclei (biology constraint)
    cell_mask_combined = np.logical_or(cell_mask_raw, nuclei_mask)
    
    # Morphological closing to fill holes inside the cell
    cell_mask_filled = binary_closing(cell_mask_combined, disk(5))
    
    # --- Watershed Segmentation ---
    # To get accurate cell boundaries corresponding to nuclei, we use watershed.
    # The "elevation map" is the inverted intensity of the cell signal.
    # Seeds are the nuclei. Mask is the thresholded cell area.
    
    # If no nuclei found, return 0
    if np.max(nuclei_labels) == 0:
        return 0.0

    # Watershed
    # We use the negative of the smoothed combined signal as the topographic map
    # So bright centers are basins
    elevation_map = -ndimage.gaussian_filter(dapi_norm + cyto_signal, sigma=2)
    
    segmentation = watershed(elevation_map, nuclei_labels, mask=cell_mask_filled)

    # --- Feature Calculation ---
    # We iterate through each segmented cell to calculate the ratio per cell, then average.
    # This is more robust to debris than global area summation.
    
    props = regionprops(segmentation, intensity_image=dapi_norm) # intensity image not strictly needed for area
    
    ratios = []
    
    for prop in props:
        # Get the label ID
        label_id = prop.label
        
        # Extract the area of the whole cell (nucleus + cytoplasm)
        total_cell_area = prop.area
        
        # Extract the area of the nucleus within this cell region
        # We look at the intersection of the specific cell label and the original nuclei mask
        # Optimization: use the bounding box of the region to slice the big mask
        minr, minc, maxr, maxc = prop.bbox
        cell_crop = (segmentation[minr:maxr, minc:maxc] == label_id)
        nuclei_crop = nuclei_mask[minr:maxr, minc:maxc]
        
        # Intersection
        nucleus_area = np.sum(cell_crop & nuclei_crop)
        
        # Cytoplasm area = Total Cell Area - Nucleus Area
        cytoplasm_area = total_cell_area - nucleus_area
        
        # Avoid division by zero
        if cytoplasm_area > 0:
            ratio = nucleus_area / cytoplasm_area
            # Filter out unreasonable ratios (e.g., artifacts where nucleus > cell or tiny cytoplasm)
            # A typical N/C ratio is < 1. Cancer cells might be higher, but > 10 is likely noise.
            if ratio < 10.0: 
                ratios.append(ratio)
    
    if not ratios:
        return 0.0
        
    # Return the median ratio to be robust against outliers
    return float(np.median(ratios))
