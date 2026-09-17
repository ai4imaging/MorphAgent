def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, remove_small_objects, disk
    from scipy import ndimage

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3) - Channel 0 is Actin
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract Actin channel (Channel 0)
        actin_channel = arr[:, :, 0]
    elif arr.ndim == 2:
        # Fallback if passed a single channel image (unlikely based on spec but safe)
        actin_channel = arr
    else:
        # Unexpected format
        return 0.0

    # Intensity normalization for the actin channel
    # Normalize to [0, 1] for thresholding stability
    vmax = np.percentile(actin_channel, 99.5) if actin_channel.size > 0 else 1.0
    if vmax > 0:
        actin_channel = actin_channel / vmax
    actin_channel = np.clip(actin_channel, 0.0, 1.0)

    # Determine the mask to use for measuring eccentricity
    labeled_mask = None

    # Check if valid segmentation masks are provided
    # We prefer a cell mask if available. Usually masks are passed as (cell_mask, nuclei_mask) or similar.
    # We iterate to find a mask that looks like a cell mask (larger objects) or just use the first one.
    if len(segmentation_masks) > 0:
        for mask in segmentation_masks:
            if mask is not None and mask.shape == actin_channel.shape:
                # If the mask is already labeled (int), use it directly
                # If it's binary, label it
                if np.issubdtype(mask.dtype, np.integer) and mask.max() > 1:
                    labeled_mask = mask
                    break
                elif mask.max() > 0:
                    labeled_mask = label(mask > 0)
                    break
    
    # If no valid mask provided, generate one from the Actin channel
    if labeled_mask is None:
        # Apply Gaussian blur to smooth texture
        smoothed = ndimage.gaussian_filter(actin_channel, sigma=2.0)
        
        # Determine threshold
        try:
            thresh = threshold_otsu(smoothed)
        except ValueError:
            # Handle case where image is uniform (e.g., all black)
            thresh = 0.5
            
        # Create binary mask
        binary_mask = smoothed > thresh
        
        # Morphological cleanup
        # Close gaps in the cytoskeleton
        binary_mask = binary_closing(binary_mask, disk(3))
        # Remove small noise (debris)
        binary_mask = remove_small_objects(binary_mask, min_size=100)
        
        # Label the objects
        labeled_mask = label(binary_mask)

    # Compute properties
    regions = regionprops(labeled_mask)
    
    if not regions:
        return 0.0

    # Extract eccentricity for all regions
    # Eccentricity is a scalar between 0 (circle) and 1 (line)
    eccentricities = [r.eccentricity for r in regions]
    
    if not eccentricities:
        return 0.0

    # Compute the mean eccentricity
    mean_eccentricity = np.mean(eccentricities)

    return float(mean_eccentricity)
