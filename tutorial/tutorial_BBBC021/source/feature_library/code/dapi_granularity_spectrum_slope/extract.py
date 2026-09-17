def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES
    import numpy as np
    from scipy import ndimage, stats, fft
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu

    # 1. Data Loading and Preprocessing
    # Convert to float32 for processing
    arr = np.asarray(img, dtype=np.float32)

    # Handle Dimensions: Expecting (H, W, C) = (512, 512, 3)
    # Channel 2 is DAPI (Nucleus)
    if arr.ndim == 3 and arr.shape[2] == 3:
        dapi_channel = arr[..., 2]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely given description but safe)
        dapi_channel = arr
    else:
        return 0.0

    # Normalize intensity to [0, 1]
    # Robust normalization using percentiles to handle outliers
    p_min, p_max = np.percentile(dapi_channel, (1, 99))
    if p_max > p_min:
        dapi_channel = (dapi_channel - p_min) / (p_max - p_min)
    else:
        dapi_channel = dapi_channel - p_min # Should be 0
    dapi_channel = np.clip(dapi_channel, 0.0, 1.0)

    # 2. Segmentation Handling
    # We need to analyze texture *within* nuclei.
    # If masks are provided, use the first one (likely nuclei).
    # If not, generate a mask using Otsu thresholding.
    mask = None
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask = segmentation_masks[0]
        # Ensure mask is 2D
        if mask.ndim == 3:
            mask = mask.max(axis=2) if mask.shape[2] < 5 else mask.max(axis=0)
    
    if mask is None:
        try:
            thresh = threshold_otsu(dapi_channel)
            mask = dapi_channel > thresh
        except Exception:
            # Fallback for completely flat images
            return 0.0

    # Label connected components (nuclei)
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask, intensity_image=dapi_channel)

    # 3. Feature Computation: Spectral Slope per Nucleus
    slopes = []
    
    # Minimum size for a meaningful FFT (e.g., 16x16 pixels)
    min_dim = 16 

    for region in regions:
        # Extract bounding box of the nucleus
        minr, minc, maxr, maxc = region.bbox
        height = maxr - minr
        width = maxc - minc

        if height < min_dim or width < min_dim:
            continue

        # Extract the patch
        patch = region.image  # Binary mask of the object in the bbox
        intensity_patch = region.intensity_image # Intensity values in the bbox
        
        # We only want the texture *inside* the nucleus.
        # However, FFT requires a rectangular grid.
        # Strategy: Use the intensity patch where mask is True, 0 elsewhere.
        # Crucial: Apply a window function (Hanning) to reduce spectral leakage from edges.
        
        # Create 2D Hanning window
        # Note: We use numpy's hanning, outer product for 2D
        win_h = np.hanning(height)
        win_w = np.hanning(width)
        window = np.outer(win_h, win_w)
        
        # Apply window to the intensity patch. 
        # We multiply by the binary patch first to zero out background in the bbox,
        # then subtract mean to remove DC component, then window.
        
        # Mask out background within the bbox
        masked_intensity = intensity_patch * patch
        
        # Calculate mean of the *object* pixels only to center the data
        if np.sum(patch) > 0:
            mean_val = np.sum(masked_intensity) / np.sum(patch)
        else:
            mean_val = 0
            
        # Subtract mean (only from object pixels) and apply window
        # Pixels outside the object (where patch is 0) remain 0 after windowing
        # because windowing tapers to 0 at edges, but we must ensure internal consistency.
        # A simpler robust approach for irregular shapes:
        # Just window the whole rectangular patch. The background is 0 (black).
        # The sharp transition from object to black background creates high freq artifacts.
        # The window function helps mitigate this by forcing the patch edges to 0.
        
        prepared_patch = (masked_intensity - mean_val) * patch * window

        # 4. Compute 2D FFT
        try:
            f_transform = fft.fft2(prepared_patch)
            f_shift = fft.fftshift(f_transform)
            magnitude_spectrum = np.abs(f_shift) ** 2
        except ValueError:
            continue

        # 5. Azimuthal Integration (Radial Profile)
        h, w = prepared_patch.shape
        center_y, center_x = h // 2, w // 2
        
        y, x = np.indices((h, w))
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        r = r.astype(int)

        # Sum power in each ring
        tbin = np.bincount(r.ravel(), weights=magnitude_spectrum.ravel())
        nr = np.bincount(r.ravel())
        
        # Avoid division by zero
        radial_profile = tbin / (nr + 1e-10)

        # 6. Calculate Slope (Log-Log)
        # Exclude DC component (index 0) and very high frequencies (noise/aliasing)
        # We typically look at the middle range.
        # Max radius is min(h, w) / 2
        max_r = min(h, w) // 2
        
        # Start from 1 to skip DC, go up to max_r
        if max_r < 3: 
            continue
            
        freqs = np.arange(1, max_r)
        power = radial_profile[1:max_r]

        # Filter out zeros or negative values (log domain issues)
        valid_mask = (power > 1e-20)
        if np.sum(valid_mask) < 3:
            continue
            
        log_freqs = np.log(freqs[valid_mask])
        log_power = np.log(power[valid_mask])

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_freqs, log_power)
        
        if not np.isnan(slope):
            slopes.append(slope)

    # 7. Aggregation
    # Return the median slope across all nuclei.
    # A steeper negative slope (more negative) -> smoother texture.
    # A flatter slope (closer to 0) -> more granular/noisy texture.
    if not slopes:
        return 0.0
        
    result = np.median(slopes)
    return float(result)
