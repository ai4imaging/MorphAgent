def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import shannon_entropy, label, regionprops
    from skimage.filters import threshold_otsu
    from scipy import ndimage

    # 1. Input Validation and Channel Extraction
    # Ensure input is a numpy array
    img = np.asarray(img)
    
    # Check dimensionality and extract DAPI channel (Channel 2)
    # Dataset description: (512, 512, 3), Channel 2 is DAPI (Blue)
    if img.ndim == 3 and img.shape[2] == 3:
        dapi_channel = img[:, :, 2]
    elif img.ndim == 2:
        # Fallback: if single channel, assume it's the relevant one or a projection
        dapi_channel = img
    else:
        # Unexpected format
        return 0.0

    # 2. Segmentation Logic
    # Determine if we have a valid segmentation mask or need to generate one
    mask = None
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the provided mask (assuming the first one is the primary/nuclear mask)
        mask = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask.shape != dapi_channel.shape:
            # If mask is 3D or different size, try to handle or discard
            if mask.ndim == 3 and mask.shape[:2] == dapi_channel.shape[:2]:
                mask = mask[:, :, 0] # Take first channel of mask if multi-channel
            elif mask.shape != dapi_channel.shape:
                mask = None # Invalid mask shape, fallback to auto

    if mask is None:
        # Fallback: Automatic segmentation on DAPI channel
        # Smooth slightly to reduce noise before thresholding
        smoothed = ndimage.gaussian_filter(dapi_channel.astype(float), sigma=2.0)
        try:
            thresh = threshold_otsu(smoothed)
            mask = smoothed > thresh
        except Exception:
            # If image is uniform (e.g. all black), otsu fails
            return 0.0

    # 3. Labeling and Region Extraction
    # Ensure mask is labeled (integers for distinct objects)
    if mask.dtype == bool:
        labeled_mask = label(mask)
    else:
        labeled_mask = mask.astype(int)

    # Get properties for each nucleus
    regions = regionprops(labeled_mask, intensity_image=dapi_channel)

    if not regions:
        return 0.0

    # 4. Feature Computation: Entropy per Nucleus
    entropies = []
    
    for region in regions:
        # Filter small artifacts
        if region.area < 50:
            continue
            
        # Extract pixel intensities within the nucleus
        # region.image_intensity gives the intensity values inside the bounding box masked by the region
        intensity_vals = region.image_intensity[region.image]
        
        # Calculate Shannon entropy
        # Entropy requires discrete bins. Since input is uint8 (0-255), we can use it directly.
        # If float, we might need to bin, but dataset is uint8.
        if intensity_vals.size > 0:
            # shannon_entropy calculates base 2 entropy of the histogram
            e = shannon_entropy(intensity_vals)
            entropies.append(e)

    # 5. Aggregation
    # Return the mean entropy across all detected nuclei
    if not entropies:
        return 0.0
        
    result = np.mean(entropies)

    return float(result)
