def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import stats
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) where Channel 2 is DAPI (Blue)
    if arr.ndim == 3 and arr.shape[2] == 3:
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if single channel passed, though dataset says 3-channel
        dapi_channel = arr
    else:
        return 0.0

    # Determine the mask for nuclei
    # Priority: Use provided segmentation masks if available, otherwise compute Otsu
    nuclei_mask = None
    
    if segmentation_masks and len(segmentation_masks) > 0:
        # Check if any mask is valid (non-None and matching shape)
        for mask in segmentation_masks:
            if mask is not None:
                # Handle potential shape mismatch if mask is 3D but image is 2D slice or vice versa
                # The dataset says masks are likely 2D labeled arrays
                if mask.shape == dapi_channel.shape:
                    nuclei_mask = mask
                    break
                elif mask.ndim == 3 and mask.shape[:2] == dapi_channel.shape:
                    # If mask has channels, take the first one or max projection
                    nuclei_mask = mask[:, :, 0]
                    break
    
    # If no valid mask provided, generate one using Otsu thresholding on DAPI
    if nuclei_mask is None:
        # Basic normalization for thresholding
        # Check if image is not empty/black
        if np.max(dapi_channel) == np.min(dapi_channel):
            return 0.0
            
        try:
            thresh = threshold_otsu(dapi_channel)
            binary_mask = dapi_channel > thresh
            nuclei_mask = label(binary_mask)
        except Exception:
            return 0.0

    # Ensure mask is labeled (integers for distinct objects)
    # If the provided mask was binary, label it now
    if nuclei_mask.max() == 1 and nuclei_mask.dtype == bool:
        nuclei_mask = label(nuclei_mask)
    elif nuclei_mask.max() == 1: # integer binary
        nuclei_mask = label(nuclei_mask)
        
    # Get properties of labeled regions
    regions = regionprops(nuclei_mask.astype(int), intensity_image=dapi_channel)
    
    if not regions:
        return 0.0

    kurtosis_values = []

    for region in regions:
        # Extract pixel intensities for the current nucleus
        intensities = region.image_intensity[region.image] # This gets values where mask is True in the bounding box
        
        # Filter out very small regions to ensure statistical stability
        if intensities.size < 10:
            continue
            
        # Calculate Kurtosis (Fisher's definition, excess kurtosis, normal = 0.0)
        # We need variance > 0 to avoid division by zero
        if np.std(intensities) > 1e-6:
            # bias=False corrects for statistical bias in sample kurtosis
            k = stats.kurtosis(intensities, fisher=True, bias=False)
            
            # Check for NaN or Inf
            if np.isfinite(k):
                kurtosis_values.append(k)

    # Return the mean kurtosis across all valid nuclei
    if not kurtosis_values:
        return 0.0
        
    result = np.mean(kurtosis_values)
    
    return float(result)
