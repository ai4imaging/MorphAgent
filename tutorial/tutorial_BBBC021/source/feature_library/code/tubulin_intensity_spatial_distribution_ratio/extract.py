def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.morphology import disk, binary_dilation, binary_erosion

    # 1. Data Loading and Validation
    # Ensure input is a numpy array
    arr = np.asarray(img)
    
    # Check dimensionality and shape
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # 2. Channel Extraction
    # Channel 0: Actin (Red) - Cytoskeleton structure
    # Channel 1: Tubulin (Green) - Target for intensity measurement
    # Channel 2: DAPI (Blue) - Nucleus (Anchor for spatial regions)
    actin_ch = arr[..., 0].astype(np.float32)
    tubulin_ch = arr[..., 1].astype(np.float32)
    dapi_ch = arr[..., 2].astype(np.float32)

    # Normalize Tubulin channel for measurement (0-1 range)
    # Using a robust max to avoid hot pixel artifacts
    tub_max = np.percentile(tubulin_ch, 99.9)
    if tub_max > 0:
        tubulin_norm = tubulin_ch / tub_max
    else:
        tubulin_norm = tubulin_ch

    # 3. Segmentation (ROI Definition)
    # We need to define: Nucleus, Perinuclear Region, and Peripheral Region.
    # Since segmentation masks might not be provided or reliable for sub-cellular zones,
    # we compute global masks on the fly.

    # A. Nuclear Mask (from DAPI)
    # Smooth slightly to reduce noise
    dapi_smooth = ndimage.gaussian_filter(dapi_ch, sigma=2)
    try:
        thresh_nuc = threshold_otsu(dapi_smooth)
        mask_nuc = dapi_smooth > thresh_nuc
    except Exception:
        # Fallback if image is empty/uniform
        return 0.0

    # B. Cell Body Mask (Union of Actin and Tubulin)
    # This defines the total cytoplasmic area
    cyto_signal = np.maximum(actin_ch, tubulin_ch)
    cyto_smooth = ndimage.gaussian_filter(cyto_signal, sigma=2)
    try:
        thresh_cyto = threshold_otsu(cyto_smooth)
        # The cell mask must include the nucleus
        mask_cell = (cyto_smooth > thresh_cyto) | mask_nuc
    except Exception:
        return 0.0

    # 4. Define Spatial Zones
    # Zone 1: Perinuclear Zone
    # Defined as a ring around the nucleus.
    # We dilate the nuclear mask by a fixed radius.
    # For 512x512 HCS images, a radius of ~15 pixels is a reasonable approximation for the perinuclear zone.
    dilation_radius = 15
    selem = disk(dilation_radius)
    
    # Dilate nuclei to create the "Nucleus + Perinuclear" area
    mask_nuc_dilated = binary_dilation(mask_nuc, footprint=selem)
    
    # The Perinuclear Zone is: (Dilated Nuclei) AND (Cell Body) AND NOT (Nuclei)
    mask_perinuclear = mask_nuc_dilated & mask_cell & (~mask_nuc)

    # Zone 2: Peripheral Zone
    # Defined as the rest of the cytoplasm outside the perinuclear zone.
    # Logic: (Cell Body) AND NOT (Dilated Nuclei)
    mask_peripheral = mask_cell & (~mask_nuc_dilated)

    # 5. Feature Calculation
    # We calculate the mean intensity of Tubulin in both zones.
    # We use a global approach (all pixels in the image) rather than per-cell,
    # which is robust against segmentation errors in dense clumps.

    pixels_peri = tubulin_norm[mask_perinuclear]
    pixels_outer = tubulin_norm[mask_peripheral]

    # Check if we have valid pixels to measure
    if pixels_peri.size == 0 or pixels_outer.size == 0:
        # If we can't define one of the zones (e.g., cell is too small or all nucleus),
        # return 1.0 (neutral) or 0.0. 
        # If no periphery exists, it implies extreme collapse or small cells -> conceptually high ratio,
        # but numerically undefined. We return 0.0 to indicate failure to extract.
        return 0.0

    mean_intensity_peri = np.mean(pixels_peri)
    mean_intensity_outer = np.mean(pixels_outer)

    # 6. Compute Ratio
    # Ratio = Perinuclear Intensity / Peripheral Intensity
    # High ratio (>1) -> Tubulin collapsed around nucleus (e.g., Vinca alkaloids)
    # Low/Unit ratio (~1) -> Tubulin spread throughout cytoplasm (Control/Taxanes)
    
    epsilon = 1e-7 # Avoid division by zero
    ratio = mean_intensity_peri / (mean_intensity_outer + epsilon)

    return float(ratio)
