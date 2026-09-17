def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import binary_dilation, disk
    from skimage.filters import threshold_otsu
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)
    
    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), uint8
    # Channel 1 is Tubulin (Green), Channel 2 is DAPI (Blue)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # Fallback or error handling for unexpected shapes
        return 0.0

    # Extract relevant channels
    # Channel 1: Tubulin (Intensity source)
    tubulin_channel = arr[..., 1]
    # Channel 2: DAPI (Segmentation source if masks not provided)
    dapi_channel = arr[..., 2]

    # Intensity normalization for Tubulin channel
    # Normalize to [0, 1] based on uint8 range
    tubulin_norm = tubulin_channel / 255.0
    tubulin_norm = np.clip(tubulin_norm, 0.0, 1.0)

    # Determine Nuclear Mask
    nuclear_mask = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Assume the first mask is the nuclear mask based on standard workflows
        # Masks are typically labeled integers (0=bg, 1..N=cells)
        seg = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if seg.shape == arr.shape[:2]:
            nuclear_mask = seg > 0
            
    # Fallback: Generate mask from DAPI channel if no external mask provided
    if nuclear_mask is None:
        # Normalize DAPI for thresholding
        dapi_norm = dapi_channel / 255.0
        # Simple background check to avoid error on empty images
        if np.max(dapi_norm) > 0.05:
            try:
                thresh = threshold_otsu(dapi_norm)
                nuclear_mask = dapi_norm > thresh
            except Exception:
                # Fallback for extremely low contrast images
                nuclear_mask = dapi_norm > 0.1
        else:
            nuclear_mask = np.zeros(dapi_channel.shape, dtype=bool)

    # If no nuclei found, return 0.0
    if not np.any(nuclear_mask):
        return 0.0

    # Define Perinuclear Region (The Ring)
    # We want a ring immediately surrounding the nucleus.
    # A radius of 5 pixels is appropriate for 512x512 MCF-7 images to capture 
    # the immediate ER/Golgi/Centrosome region without extending too far.
    ring_width = 5
    selem = disk(ring_width)
    
    # Create the dilated mask (Nucleus + Ring)
    dilated_mask = binary_dilation(nuclear_mask, selem)
    
    # Create the ring mask: (Dilated) MINUS (Original Nucleus)
    # This gives us the donut shape.
    ring_mask = dilated_mask & (~nuclear_mask)
    
    # Refinement: In dense cell clusters, the ring of one cell might overlap 
    # with the nucleus of a neighbor. We should exclude ANY nuclear pixels 
    # from the ring mask to ensure we are measuring cytoplasm/perinuclear space.
    # Since 'nuclear_mask' contains all nuclei, the operation above 
    # (dilated & ~nuclear_mask) already handles this correctly for the global mask.
    
    # Extract pixels corresponding to the perinuclear rings
    perinuclear_pixels = tubulin_norm[ring_mask]
    
    # Compute the feature: Mean Intensity
    if perinuclear_pixels.size == 0:
        result = 0.0
    else:
        result = np.mean(perinuclear_pixels)

    return float(result)
