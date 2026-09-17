def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_dilation, disk

    # Convert to float32 for processing
    arr = np.asarray(img, dtype=np.float32)

    # 1. Data Validation and Preprocessing
    # Check shape: Expecting (H, W, 3) for BBBC021
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Normalize intensity to [0, 1]
    # Using a robust max to avoid hot pixel scaling issues
    vmax = np.percentile(arr, 99.9) if arr.size > 0 else 1.0
    if vmax > 0:
        arr = arr / vmax
    arr = np.clip(arr, 0.0, 1.0)

    # Extract channels based on dataset description
    # Channel 0: Actin (Red) - Cytoskeleton
    # Channel 1: Tubulin (Green) - Microtubules (Target Signal)
    # Channel 2: DAPI (Blue) - Nucleus (Anchor)
    actin = arr[..., 0]
    tubulin = arr[..., 1]
    dapi = arr[..., 2]

    # 2. Segmentation Logic
    # We need to define three regions: Nucleus, Perinuclear Cytoplasm, Peripheral Cytoplasm.
    # Since segmentation masks are optional/variable, we implement a robust fallback
    # using the DAPI and Actin channels which are high-contrast in this dataset.

    # A. Nuclei Segmentation (Anchor)
    # Smooth DAPI to reduce noise
    dapi_smooth = ndimage.gaussian_filter(dapi, sigma=2.0)
    try:
        thresh_nuc = threshold_otsu(dapi_smooth)
    except Exception:
        return 0.0 # Empty image or uniform background

    mask_nuc = dapi_smooth > thresh_nuc
    # Fill holes to make solid objects
    mask_nuc = ndimage.binary_fill_holes(mask_nuc)

    # B. Cell Body Segmentation (Boundary)
    # Combine Actin and Tubulin for best cell outline
    cell_signal = np.maximum(actin, tubulin)
    cell_smooth = ndimage.gaussian_filter(cell_signal, sigma=2.0)
    try:
        thresh_cell = threshold_otsu(cell_smooth)
    except Exception:
        # If cell threshold fails, we can't define cytoplasm
        return 0.0
    
    mask_cell = cell_smooth > thresh_cell
    mask_cell = ndimage.binary_fill_holes(mask_cell)

    # C. Cytoplasm Definition
    # Cytoplasm is Cell Body minus Nucleus
    mask_cyto = np.logical_and(mask_cell, ~mask_nuc)

    # Check if we have any cytoplasm to analyze
    if np.sum(mask_cyto) == 0:
        return 0.0

    # 3. Radial Zoning (Perinuclear vs Peripheral)
    # We define the "Perinuclear" region as the cytoplasm immediately surrounding the nucleus.
    # We define "Peripheral" as the rest of the cytoplasm.
    
    # Method: Distance Transform
    # Calculate Euclidean distance from the nearest nuclear pixel for every pixel in the image.
    # We invert mask_nuc because distance_transform_edt calculates distance to the nearest zero.
    dist_from_nuc = ndimage.distance_transform_edt(~mask_nuc)

    # Define the Perinuclear Zone width (in pixels)
    # For 512x512 images of MCF-7 cells, a band of ~15-20 pixels (approx 5-10 microns depending on resolution)
    # captures the centrosomal region where monoastral spindles concentrate.
    perinuclear_width = 15.0

    # Create masks for the two zones, constrained to the valid cytoplasm mask
    # Zone 1: Inner (Perinuclear) -> Distance > 0 (outside nucleus) AND Distance <= width
    mask_inner = np.logical_and(mask_cyto, dist_from_nuc <= perinuclear_width)
    
    # Zone 2: Outer (Peripheral) -> Distance > width
    mask_outer = np.logical_and(mask_cyto, dist_from_nuc > perinuclear_width)

    # 4. Feature Calculation: Tubulin Concentration Ratio
    # We calculate the mean intensity of Tubulin in both zones.
    
    # Extract pixels
    inner_pixels = tubulin[mask_inner]
    outer_pixels = tubulin[mask_outer]

    # Handle edge cases where zones might be empty
    if inner_pixels.size == 0:
        # No perinuclear region? Unlikely if cyto exists, but implies no signal near nucleus.
        return 0.0
    
    mean_inner = np.mean(inner_pixels)

    if outer_pixels.size == 0:
        # No peripheral region? This happens if the cell is very small/round (e.g., extreme rounding).
        # In this case, the concentration is effectively infinite or maximal.
        # We return a high value to indicate high central concentration.
        # However, to keep it numerical, we divide by a small epsilon.
        mean_outer = 0.0
    else:
        mean_outer = np.mean(outer_pixels)

    # Compute Ratio
    # Add small epsilon to denominator to avoid division by zero
    epsilon = 1e-6
    ratio = mean_inner / (mean_outer + epsilon)

    return float(ratio)
