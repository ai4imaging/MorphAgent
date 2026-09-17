def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    
    # --- 1. Data Loading and Preprocessing ---
    # Convert to float32 for calculations
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensionality and extract Actin channel (Channel 0)
    # Expected shape: (512, 512, 3)
    if arr.ndim == 3 and arr.shape[2] >= 1:
        actin_img = arr[..., 0]  # Channel 0 is Actin
    elif arr.ndim == 2:
        actin_img = arr  # Fallback if single channel
    else:
        return 0.0

    # Normalize intensity to [0, 1] for stability
    vmax = np.percentile(actin_img, 99.5) if actin_img.size > 0 else 1.0
    if vmax > 0:
        actin_img = actin_img / vmax
    actin_img = np.clip(actin_img, 0.0, 1.0)

    # --- 2. Mask Handling ---
    # We need a cell mask to define the boundary.
    # If segmentation masks are provided, we look for a cell mask.
    # Usually, if multiple masks are provided, one is nuclei and one is cells/cytoplasm.
    # We prioritize the largest mask coverage as the "cell" mask.
    
    cell_mask = None
    
    if len(segmentation_masks) > 0:
        # Heuristic: The mask with the largest foreground area is likely the whole-cell mask
        # (as opposed to a nuclei mask which is smaller).
        best_coverage = -1
        for mask in segmentation_masks:
            if mask is None: continue
            # Ensure mask is 2D
            if mask.ndim == 3:
                mask = np.max(mask, axis=2) # Project if 3D
            
            coverage = np.sum(mask > 0)
            if coverage > best_coverage:
                best_coverage = coverage
                cell_mask = mask
    
    # Fallback: If no mask provided, generate one using Otsu thresholding on Actin
    if cell_mask is None:
        try:
            # Smooth slightly before thresholding
            smooth = ndimage.gaussian_filter(actin_img, sigma=2)
            thresh = threshold_otsu(smooth)
            cell_mask = (smooth > thresh).astype(np.int32)
            # Label connected components
            cell_mask = label(cell_mask)
        except Exception:
            return 0.0

    # Ensure cell_mask is labeled (integers > 0 for objects)
    if cell_mask.max() == 0:
        return 0.0
        
    # If mask is binary (0/1), label it
    if cell_mask.max() == 1:
        cell_mask = label(cell_mask)

    # --- 3. Feature Computation: Cortical Enrichment ---
    # Algorithm:
    # For each cell:
    #   1. Define "Cortex" as the outer boundary ring (e.g., 3-5 pixels thick).
    #   2. Define "Interior" as the remaining inner part.
    #   3. Calculate Mean Intensity (Cortex) / Mean Intensity (Interior).
    #   4. Aggregate across all cells (Median).

    cortex_width = 4  # Width of the cortical ring in pixels
    ratios = []

    # Get properties for each labeled cell
    props = regionprops(cell_mask, intensity_image=actin_img)

    for prop in props:
        # Skip very small artifacts
        if prop.area < 100:
            continue

        # Extract the local mask and intensity image for this cell
        # image_intensity is the crop of the intensity image
        # image is the crop of the binary mask
        local_mask = prop.image  # Binary mask of the cell in the bounding box
        local_intensity = prop.image_intensity

        # Create the Interior mask by eroding the cell mask
        # We use binary_erosion. The structure determines connectivity.
        # Iterations determine the thickness of the cortex.
        eroded_mask = ndimage.binary_erosion(local_mask, structure=np.ones((3,3)), iterations=cortex_width)
        
        # The Cortex is the Cell Mask minus the Eroded Mask
        # XOR works because eroded is a strict subset of local_mask
        cortex_mask = np.logical_xor(local_mask, eroded_mask)
        
        # Interior is the eroded mask
        interior_mask = eroded_mask

        # Check if we have valid regions (small cells might disappear after erosion)
        if np.sum(cortex_mask) == 0 or np.sum(interior_mask) == 0:
            continue

        # Calculate Mean Intensities
        mean_cortex = np.mean(local_intensity[cortex_mask])
        mean_interior = np.mean(local_intensity[interior_mask])

        # Calculate Ratio
        # Add epsilon to denominator to avoid division by zero
        ratio = mean_cortex / (mean_interior + 1e-7)
        ratios.append(ratio)

    # --- 4. Aggregation ---
    if not ratios:
        return 0.0

    # Return the median ratio across the population
    # Median is robust to segmentation errors or debris
    result = np.median(ratios)

    return float(result)
