def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import shannon_entropy, label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import closing, square
    from scipy import ndimage

    # 1. Input Validation and Channel Extraction
    # The dataset description specifies (512, 512, 3) images.
    # Channel 2 (Blue) is DAPI (Nucleus).
    if img is None:
        return 0.0
    
    # Handle dimensionality
    if img.ndim == 3 and img.shape[2] == 3:
        # Extract DAPI channel (Channel 2)
        dapi_channel = img[:, :, 2]
    elif img.ndim == 2:
        # If passed a single channel 2D image, assume it's the relevant one
        dapi_channel = img
    else:
        # Unexpected format
        return 0.0

    # Ensure dapi_channel is uint8 for discrete entropy calculation
    # If it's float, we need to bin it, but the dataset says uint8.
    # If it was normalized previously, we might need to rescale, but here we assume raw input.
    if dapi_channel.dtype != np.uint8:
        # Normalize to 0-255 if not already
        if dapi_channel.max() <= 1.0:
            dapi_channel = (dapi_channel * 255).astype(np.uint8)
        else:
            dapi_channel = dapi_channel.astype(np.uint8)

    # 2. Mask Handling
    # We need a mask identifying the nuclei.
    mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Assume the first mask is the nuclear mask or cell mask
        # If it's a labeled mask (int), convert to boolean for masking pixels
        # If it's a boolean mask, use directly
        input_mask = segmentation_masks[0]
        if input_mask.shape == dapi_channel.shape:
            mask = input_mask > 0
    
    # Fallback: Generate mask if none provided
    if mask is None:
        # Simple Otsu thresholding on the DAPI channel
        # Smooth slightly to reduce noise
        blurred = ndimage.gaussian_filter(dapi_channel, sigma=2.0)
        try:
            thresh = threshold_otsu(blurred)
            mask = blurred > thresh
            # Fill holes to ensure solid nuclei
            mask = closing(mask, square(3))
        except Exception:
            # Fallback if image is uniform (e.g. all black)
            return 0.0

    # 3. Feature Computation: Nucleus Intensity Entropy
    # We want the entropy of the pixel intensities *within* the nuclei.
    # We compute this per nucleus to be robust to cell count, then average.
    
    # Label the mask to identify individual nuclei
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask, intensity_image=dapi_channel)

    if not regions:
        return 0.0

    entropies = []
    for region in regions:
        # Filter out very small artifacts
        if region.area < 10:
            continue
            
        # Extract pixel intensities for this specific nucleus
        # region.image is the binary mask of the object in the bounding box
        # region.intensity_image is the intensity image in the bounding box
        # We select pixels where the mask is True
        nucleus_pixels = region.intensity_image[region.image]
        
        if nucleus_pixels.size == 0:
            continue

        # Compute Shannon entropy
        # shannon_entropy in skimage uses base 2 by default
        # It computes the histogram of the input array
        val = shannon_entropy(nucleus_pixels)
        entropies.append(val)

    # 4. Aggregation
    # Return the mean entropy across all detected nuclei
    if not entropies:
        return 0.0
        
    result = np.mean(entropies)

    return float(result)
