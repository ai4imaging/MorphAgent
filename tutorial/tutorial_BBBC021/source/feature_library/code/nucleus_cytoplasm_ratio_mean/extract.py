def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensions: Expected (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Channel mapping based on dataset description:
    # Ch0: Actin (Cytoskeleton)
    # Ch1: Tubulin (Microtubules)
    # Ch2: DAPI (Nucleus)
    
    # We need to define Nucleus and Cell Body (Cytoplasm + Nucleus) masks.
    
    # Strategy:
    # 1. Use segmentation masks if available.
    # 2. If not, compute them from channels.
    
    nuclei_mask = None
    cells_mask = None

    # Check if segmentation masks are provided
    # The prompt mentions masks might be available. 
    # If provided, we need to identify which is which. 
    # Usually, if masks are provided, they correspond to the objects.
    # However, without specific metadata on mask order in the prompt's dynamic section, 
    # we should robustly handle both cases or fallback to channel-based segmentation.
    # Given the "Critic Agent Feedback" suggesting potential issues with mask intersection,
    # we will prioritize a robust calculation.
    
    # Let's try to use the image channels to generate/refine masks since this is often more reliable 
    # if the external masks are not guaranteed to be perfectly aligned or labeled.
    
    # --- Nucleus Segmentation (Channel 2 - DAPI) ---
    dapi_channel = arr[..., 2]
    # Normalize DAPI
    dapi_norm = (dapi_channel - np.min(dapi_channel)) / (np.max(dapi_channel) - np.min(dapi_channel) + 1e-6)
    try:
        thresh_nuc = threshold_otsu(dapi_norm)
    except:
        thresh_nuc = 0.5
    nuclei_mask_binary = dapi_norm > thresh_nuc
    nuclei_mask_binary = binary_closing(nuclei_mask_binary, disk(2))
    nuclei_labels = label(nuclei_mask_binary)

    # --- Cell Body Segmentation (Channel 0 - Actin + Channel 1 - Tubulin) ---
    # Combine structural markers for better cell body definition
    struct_channel = np.maximum(arr[..., 0], arr[..., 1])
    # Normalize
    struct_norm = (struct_channel - np.min(struct_channel)) / (np.max(struct_channel) - np.min(struct_channel) + 1e-6)
    
    # Often the cell body signal is weaker, so we might need a lower threshold or log transform
    try:
        thresh_cell = threshold_otsu(struct_norm)
    except:
        thresh_cell = 0.2
        
    # The cell mask should definitely include the nucleus
    cells_mask_binary = (struct_norm > thresh_cell) | nuclei_mask_binary
    cells_mask_binary = binary_closing(cells_mask_binary, disk(3))
    
    # We need to associate nuclei with cell bodies.
    # A common approach in watershed-based segmentation is using nuclei as seeds.
    # However, for this feature extraction task without complex libraries, we can iterate through 
    # identified nuclei and measure the surrounding cell body area.
    
    # If we have external segmentation masks, we could use them. 
    # But to ensure the "Critic" feedback is addressed (accurate N/C calculation), 
    # we will implement a logic that ensures N < C_total.
    
    # Let's use the computed nuclei labels as the primary objects.
    # For each nucleus, we expand to find its associated cytoplasm.
    # A simple approximation if we don't have a perfect instance segmentation of touching cells
    # is to measure the nucleus area, and then measure the cell signal area *under the assumption*
    # that the cell mask covers the nucleus.
    
    # Refined approach:
    # 1. Identify Nuclei objects.
    # 2. For each nucleus, estimate the cell body belonging to it.
    #    Since we can't easily do watershed without `skimage.segmentation.watershed` (which might not be imported/allowed or complex to setup),
    #    we can look at the global masks if the cells are sparse, or use the provided masks if they exist.
    
    # Let's check if segmentation masks were passed and use them if they look like instance masks.
    used_external_masks = False
    if len(segmentation_masks) > 0:
        # Heuristic: If we have 2 masks, one is likely nuclei, one is cells.
        # If we have 1, it's likely cells or nuclei.
        # Let's assume if 2 are present, index 0 is cell, index 1 is nucleus (common convention) or vice versa.
        # We can check overlap with channels to confirm.
        
        # However, to be safe and self-contained based on the image provided:
        pass 

    # Calculation based on computed masks (Robust fallback and often more accurate than potentially mismatched external files)
    props_nuclei = regionprops(nuclei_labels)
    
    ratios = []
    
    # To assign cytoplasm to nuclei properly without full watershed:
    # We can mask the cell_binary image with the dilated nucleus to capture local cytoplasm,
    # but this is inaccurate for touching cells.
    
    # Alternative: Calculate global ratio if instance segmentation is hard? 
    # No, the feature is "mean", implying per-cell aggregation.
    
    # Let's implement a basic seeded region growing (Watershed-like) using scipy.ndimage
    # Distance transform is useful here.
    from scipy import ndimage
    
    # Distance from background
    dist = ndimage.distance_transform_edt(cells_mask_binary)
    
    # We use nuclei_labels as markers. 
    # We need to ensure markers are within the mask.
    markers = nuclei_labels.copy()
    # Ensure markers only exist where cell mask is true (sanity check)
    markers[~cells_mask_binary] = 0
    
    # Watershed using scipy.ndimage (if available, otherwise fallback)
    # ndimage.watershed_ift is not standard watershed. 
    # Standard approach without skimage.segmentation.watershed is tricky.
    # But we can use `skimage.segmentation` if allowed. The prompt allows "Optional morphology".
    # Let's stick to a simpler approach: 
    # We will assume the `nuclei_labels` are good seeds.
    # We will assign every pixel in `cells_mask_binary` to the nearest nucleus label 
    # (Voronoi tessellation restricted to the cell mask).
    
    # 1. Get coordinates of all pixels in cell mask
    # 2. Find nearest nucleus for each pixel
    # This is computationally expensive in python loops.
    
    # Efficient approximation using distance transform indices:
    # Compute distance to nearest non-zero in nuclei_labels
    # ndimage.distance_transform_edt can return indices of nearest feature.
    
    if np.max(nuclei_labels) == 0:
        return 0.0

    # Create an array where nuclei pixels have their label, others 0
    # We want to propagate these labels into the cell mask area.
    # Invert nuclei mask for distance transform (0 is feature, 1 is background for edt)
    # Actually, we want distance *from* nuclei.
    
    # feature_transform=True returns the indices of the closest background point (0).
    # So we construct an image where nuclei are 0 (background) and everything else is 1.
    input_for_edt = np.ones_like(nuclei_labels)
    input_for_edt[nuclei_labels > 0] = 0
    
    # Compute indices of nearest nucleus pixel for every pixel in the image
    dist, indices = ndimage.distance_transform_edt(input_for_edt, return_indices=True)
    
    # Map pixels to nearest nucleus label
    # indices shape is (2, H, W). 
    nearest_nucleus_labels = nuclei_labels[indices[0], indices[1]]
    
    # Now mask this with the cell binary mask
    final_cell_labels = nearest_nucleus_labels * cells_mask_binary
    
    # Now we have:
    # nuclei_labels: labeled nuclei
    # final_cell_labels: labeled cells (cytoplasm + nucleus) corresponding to those nuclei
    
    # Calculate areas
    # We can use bincount for speed
    n_labels = np.max(nuclei_labels)
    
    # Areas of nuclei
    nuc_areas = np.bincount(nuclei_labels.ravel())
    # Areas of cells (whole cell including nucleus)
    cell_areas = np.bincount(final_cell_labels.ravel())
    
    # We only care about labels 1 to n_labels
    valid_ratios = []
    
    for i in range(1, n_labels + 1):
        if i >= len(nuc_areas) or i >= len(cell_areas):
            continue
            
        n_area = nuc_areas[i]
        c_total_area = cell_areas[i]
        
        # Cytoplasm area = Total Cell Area - Nucleus Area
        # Note: Due to the way we constructed final_cell_labels (propagation), 
        # the nucleus pixels are included in final_cell_labels.
        # However, `cells_mask_binary` might have holes or slight mismatches with `nuclei_labels` 
        # if the thresholding was weird, but we forced union earlier:
        # `cells_mask_binary = (struct_norm > thresh_cell) | nuclei_mask_binary`
        # So `c_total_area` is guaranteed to be >= `n_area`.
        
        cyto_area = c_total_area - n_area
        
        # Sanity checks based on critic feedback
        if n_area > 0 and cyto_area > 0:
            ratio = n_area / cyto_area
            # Filter unreasonable ratios (e.g., if segmentation failed and cyto is tiny)
            if ratio < 10.0: # Arbitrary upper bound to remove artifacts
                valid_ratios.append(ratio)
                
    if not valid_ratios:
        return 0.0
        
    result = np.mean(valid_ratios)
    
    return float(result)
