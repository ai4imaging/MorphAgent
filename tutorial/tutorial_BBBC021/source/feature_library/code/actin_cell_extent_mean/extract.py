def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects
    from scipy import ndimage

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), Channel 0 = Actin
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Extract Actin channel (Channel 0)
        actin_channel = arr[:, :, 0]
    elif arr.ndim == 2:
        # If already 2D, assume it's the relevant channel or a projection
        actin_channel = arr
    else:
        # Unexpected format
        return 0.0

    # Normalize intensity for fallback segmentation if needed
    # Range is typically 0-255 for uint8
    if actin_channel.max() > 0:
        actin_channel = actin_channel / actin_channel.max()

    # Determine the labeled mask
    labeled_mask = None

    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask
        mask_input = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == actin_channel.shape[:2]:
            # If the mask is already labeled (integers > 1), use it directly
            if mask_input.max() > 1:
                labeled_mask = mask_input.astype(int)
            else:
                # If binary (0/1), label connected components
                labeled_mask = label(mask_input > 0)
    
    # Fallback: Generate segmentation from Actin channel if no mask provided
    if labeled_mask is None:
        try:
            # Smooth to reduce noise
            blurred = ndimage.gaussian_filter(actin_channel, sigma=2)
            
            # Threshold (Otsu)
            thresh = threshold_otsu(blurred)
            binary_mask = blurred > thresh
            
            # Clean up small artifacts (e.g., < 50 pixels)
            binary_mask = remove_small_objects(binary_mask, min_size=50)
            
            # Label connected components
            labeled_mask = label(binary_mask)
        except Exception:
            # If thresholding fails (e.g., empty image), return 0.0
            return 0.0

    # Compute Region Properties
    # We only need the label image for 'extent'
    props = regionprops(labeled_mask)

    if not props:
        return 0.0

    # Extract 'extent' for all objects
    # Extent = Area / BoundingBoxArea
    extents = [prop.extent for prop in props]

    # Calculate mean extent
    if len(extents) == 0:
        return 0.0
        
    result = np.mean(extents)

    return float(result)
