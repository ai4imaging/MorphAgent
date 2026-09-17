def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.feature import peak_local_max
    from skimage.measure import regionprops
    
    # Convert to appropriate array type
    # Image is (512, 512, 3), uint8. Channel 2 is DAPI.
    # We need the DAPI channel for this feature.
    
    # Handle dimensionality and channel selection
    # Dataset info: (Height, Width, Channels) = (512, 512, 3)
    # Channel 2 = Blue = DAPI
    if img.ndim == 3 and img.shape[2] >= 3:
        dapi_channel = img[:, :, 2]
    elif img.ndim == 2:
        # Fallback if only one channel is passed (unlikely based on spec but safe)
        dapi_channel = img
    else:
        return 0.0

    # Convert to float for processing
    dapi_float = dapi_channel.astype(np.float32)

    # Check for segmentation masks
    if not segmentation_masks:
        # Without segmentation, we cannot compute "per nucleus" statistics accurately.
        # We could try to compute it globally, but the feature is defined "within each nucleus".
        # Returning 0.0 is a safe fallback indicating no features extracted.
        return 0.0
    
    # Assume the first mask is the nuclei mask based on typical ordering
    nuclei_mask = segmentation_masks[0]
    
    # Ensure mask shape matches image shape (handle potential 2D vs 3D issues)
    if nuclei_mask.shape != dapi_channel.shape:
        return 0.0

    # Pre-processing: Smooth the image slightly to remove pixel-level noise
    # This prevents finding thousands of peaks in noisy flat regions
    # Sigma=1.0 is a standard choice for cellular microscopy at this resolution
    smoothed_dapi = ndimage.gaussian_filter(dapi_float, sigma=1.0)

    # Get properties of each labeled nucleus
    regions = regionprops(nuclei_mask, intensity_image=smoothed_dapi)
    
    if not regions:
        return 0.0

    peak_counts = []

    # Parameters for peak finding
    # min_distance: Minimum number of pixels separating peaks. 
    # For 512x512 images of cells, heterochromatin spots are usually > 2-3 pixels apart.
    min_dist = 3
    
    # threshold_abs: Minimum intensity to be considered a peak.
    # Image is uint8 (0-255). Background is usually < 20.
    # We want to count bright spots inside the nucleus.
    # We use a relative threshold based on the image content to be robust to exposure differences.
    # However, to keep it simple and robust per cell, we can use a low absolute threshold 
    # to just filter out very dark background if the mask is imperfect.
    # A better approach for "peaks" is often relative to the local region, 
    # but peak_local_max handles local maxima. We just need to ensure we don't pick up 
    # noise in dark areas.
    abs_thresh = 20.0 

    for props in regions:
        # Extract the intensity image for this specific nucleus
        # props.intensity_image is the crop of the intensity image
        # props.image is the binary mask of the nucleus within that crop
        
        # We only want peaks that are actually INSIDE the nucleus mask, 
        # not just in the bounding box.
        # Mask the intensity crop:
        roi_intensity = props.intensity_image
        roi_mask = props.image
        
        # If the region is too small, skip
        if roi_intensity.size < min_dist * min_dist:
            peak_counts.append(0)
            continue

        # Find peaks in the ROI
        # We use the ROI intensity directly. 
        # Note: peak_local_max returns coordinates relative to the ROI.
        peaks = peak_local_max(
            roi_intensity, 
            min_distance=min_dist,
            threshold_abs=abs_thresh,
            exclude_border=False,
            labels=roi_mask # Restrict peaks to the masked region
        )
        
        peak_counts.append(len(peaks))

    if not peak_counts:
        return 0.0

    # Compute the mean number of peaks per nucleus
    result = np.mean(peak_counts)

    return float(result)
