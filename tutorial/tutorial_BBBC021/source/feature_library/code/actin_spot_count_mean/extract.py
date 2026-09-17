def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import blob_log
    from skimage.filters import threshold_otsu
    from skimage.measure import label
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), uint8
    # Channel 0: Actin (Red) -> Target for spots
    # Channel 2: DAPI (Blue) -> Target for cell counting (fallback)
    
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract channels
    actin_ch = arr[:, :, 0]
    dapi_ch = arr[:, :, 2]

    # Normalize intensity to [0, 1] for consistent blob detection thresholds
    # Using simple min-max normalization based on dtype range (0-255) is safer for absolute thresholds
    # but percentile normalization is robust to outliers.
    # Here we normalize by 255.0 to keep the physical meaning of intensity consistent.
    actin_norm = actin_ch / 255.0
    dapi_norm = dapi_ch / 255.0

    # Determine Number of Cells (Denominator) and Foreground Mask
    num_cells = 0
    foreground_mask = None

    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use provided segmentation mask
        seg_mask = segmentation_masks[0]
        # Ensure mask is 2D
        if seg_mask.ndim == 3:
            seg_mask = seg_mask.max(axis=2) # Project if 3D
        elif seg_mask.ndim > 3:
            seg_mask = np.squeeze(seg_mask)
            
        # Check shape match
        if seg_mask.shape == actin_norm.shape:
            # Count unique labels (excluding 0 background)
            labels = np.unique(seg_mask)
            num_cells = len(labels) - (1 if 0 in labels else 0)
            foreground_mask = seg_mask > 0
    
    # Fallback if no valid mask or 0 cells found in mask
    if num_cells == 0:
        # Estimate cell count from DAPI channel
        try:
            thresh = threshold_otsu(dapi_norm)
            nuclei_mask = dapi_norm > thresh
            labeled_nuclei = label(nuclei_mask)
            num_cells = labeled_nuclei.max()
            
            # Create a rough foreground mask from Actin to avoid background noise
            # If actin is very faint, this might be empty, so we fallback to full image if needed
            try:
                actin_thresh = threshold_otsu(actin_norm)
                foreground_mask = actin_norm > actin_thresh
            except Exception:
                foreground_mask = np.ones_like(actin_norm, dtype=bool)
                
        except Exception:
            # If Otsu fails (e.g. uniform image), assume 1 cell or 0
            num_cells = 1
            foreground_mask = np.ones_like(actin_norm, dtype=bool)

    if num_cells == 0:
        return 0.0

    # Detect Actin Spots (Numerator)
    # Laplacian of Gaussian (LoG) is good for blob detection
    # Parameters tuned for typical focal adhesion/aggregate sizes in 512x512 microscopy images
    # min_sigma=1: small puncta
    # max_sigma=4: avoid detecting large diffuse regions
    # threshold=0.05: relative intensity threshold (since we normalized to 0-1)
    try:
        blobs = blob_log(actin_norm, min_sigma=1, max_sigma=4, num_sigma=10, threshold=0.05)
    except Exception:
        return 0.0

    if len(blobs) == 0:
        return 0.0

    # Filter spots: only count those inside the foreground mask
    # blobs returns [y, x, sigma]
    spot_count = 0
    if foreground_mask is not None:
        H, W = foreground_mask.shape
        for blob in blobs:
            y, x = int(blob[0]), int(blob[1])
            # Boundary check
            if 0 <= y < H and 0 <= x < W:
                if foreground_mask[y, x]:
                    spot_count += 1
    else:
        spot_count = len(blobs)

    # Compute Mean Spots per Cell
    result = spot_count / float(num_cells)

    return float(result)
