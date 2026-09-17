def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import stats
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from scipy import ndimage

    # 1. Data Loading and Validation
    # Ensure input is a numpy array
    img = np.asarray(img)
    
    # Check dimensionality and extract DAPI channel
    # Dataset spec: (512, 512, 3), Channel 2 is DAPI (Blue)
    if img.ndim == 3 and img.shape[2] >= 3:
        dapi_channel = img[:, :, 2]
    elif img.ndim == 2:
        # Fallback: if 2D, assume it's the relevant channel or a projection
        dapi_channel = img
    else:
        return 0.0

    # Ensure dapi_channel is uint8 for consistent histogram binning (0-255)
    # If not, normalize and convert
    if dapi_channel.dtype != np.uint8:
        # Normalize to 0-255 range robustly
        min_val = np.min(dapi_channel)
        max_val = np.max(dapi_channel)
        if max_val > min_val:
            dapi_channel = ((dapi_channel - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            dapi_channel = np.zeros_like(dapi_channel, dtype=np.uint8)

    # 2. Mask Handling
    # We need a labeled mask for nuclei to compute entropy per nucleus
    nuclei_mask = None

    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Usually, if multiple masks exist, we need to identify the nuclear one.
        # Without specific metadata, we check the masks.
        # Often, nuclear masks are smaller or the first one provided.
        # Here we take the first available mask as the primary ROI mask.
        candidate_mask = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if candidate_mask.shape == dapi_channel.shape:
            nuclei_mask = candidate_mask
        elif candidate_mask.ndim == 3 and candidate_mask.shape[:2] == dapi_channel.shape:
             # If mask is 3D (e.g. one-hot or labeled stack), take max projection or first slice
             nuclei_mask = np.max(candidate_mask, axis=2)
    
    # Fallback: Generate mask if none provided or invalid
    if nuclei_mask is None:
        # Simple segmentation pipeline
        # 1. Blur to reduce noise
        blurred = ndimage.gaussian_filter(dapi_channel, sigma=2)
        # 2. Threshold (Otsu)
        try:
            thresh = threshold_otsu(blurred)
            binary_mask = blurred > thresh
        except ValueError: # Handle empty/uniform images
            return 0.0
        # 3. Label connected components
        nuclei_mask = label(binary_mask)
    else:
        # Ensure mask is labeled (integers), not just binary
        if nuclei_mask.max() == 1 and nuclei_mask.dtype == bool:
            nuclei_mask = label(nuclei_mask)
        elif nuclei_mask.ndim > 2:
             # Handle potential extra dims in mask
             nuclei_mask = label(nuclei_mask[:,:,0] > 0)

    # 3. Feature Computation: DAPI Intensity Entropy
    # We compute the entropy of pixel intensities within each nucleus, then average across the population.
    
    entropies = []
    
    # Get properties of labeled regions
    regions = regionprops(nuclei_mask.astype(int), intensity_image=dapi_channel)
    
    for region in regions:
        # Get intensity values for the current nucleus
        intensities = region.image_intensity[region.image] # region.image is the binary mask of the bbox
        
        if intensities.size == 0:
            continue
            
        # Compute histogram of intensities
        # Using 256 bins for uint8 data [0, 255]
        counts, _ = np.histogram(intensities, bins=256, range=(0, 256))
        
        # Normalize to get probability distribution
        total_pixels = intensities.size
        if total_pixels > 0:
            probs = counts / total_pixels
            
            # Compute Shannon entropy
            # scipy.stats.entropy calculates S = -sum(pk * log(pk), axis=0)
            # base=2 is standard for bits, but natural log (base=e) is default in scipy.
            # We use base=2 for standard "bits" interpretation of texture entropy.
            ent = stats.entropy(probs, base=2)
            
            # Handle potential NaN/Inf (though stats.entropy handles 0 probs correctly)
            if np.isfinite(ent):
                entropies.append(ent)

    # 4. Aggregation
    if not entropies:
        return 0.0
        
    # Return the mean entropy across all detected nuclei
    result = np.mean(entropies)

    return float(result)
