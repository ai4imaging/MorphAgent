def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import clear_border
    from skimage.morphology import binary_dilation, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract relevant channels
    # Channel 1: Tubulin (Green) - Signal for weighted center of mass
    # Channel 2: DAPI (Blue) - Reference for nucleus center
    tubulin_ch = arr[:, :, 1]
    dapi_ch = arr[:, :, 2]

    # Intensity normalization (robust min-max)
    def normalize(channel):
        p_low, p_high = np.percentile(channel, (1, 99))
        if p_high > p_low:
            return np.clip((channel - p_low) / (p_high - p_low), 0, 1)
        return np.zeros_like(channel)

    tubulin_norm = normalize(tubulin_ch)
    dapi_norm = normalize(dapi_ch)

    # Determine Segmentation
    # We need to identify individual cells to calculate per-cell polarity.
    # The polarity is the distance between the Nucleus Center and the Tubulin Center of Mass.
    
    labeled_mask = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape == arr.shape[:2]:
            labeled_mask = mask_input.astype(int)
    
    # Fallback: Generate segmentation from DAPI channel if no mask provided
    if labeled_mask is None:
        # Simple nuclei segmentation
        thresh = threshold_otsu(dapi_norm) if np.max(dapi_norm) > 0 else 0.5
        binary_mask = dapi_norm > thresh
        # Remove artifacts
        binary_mask = ndimage.binary_opening(binary_mask, structure=np.ones((3,3)))
        # Label nuclei
        labeled_nuclei = label(binary_mask)
        # Remove border cells to avoid artificial polarity due to cropping
        labeled_mask = clear_border(labeled_nuclei)
        
        # Since we only have nuclei, we need to approximate the cell body for tubulin measurement.
        # We will use a Voronoi-like expansion (watershed) or simply dilate the nuclei 
        # to capture the perinuclear tubulin network.
        # Here, we use a simple dilation for robustness without complex watershed logic.
        # However, to avoid merging labels, we iterate or use a method that preserves labels.
        # For this feature, measuring tubulin *near* the nucleus is often sufficient, 
        # but let's try to capture a bit more context.
        # A simple approach: measure polarity on the nucleus mask itself + a small dilation.
        # Better approach: Use the labeled nuclei as the mask. The tubulin distribution *relative*
        # to the nucleus center is what matters. If the mask is too small, we miss the tails.
        # Let's expand the mask slightly to capture immediate perinuclear tubulin.
        # Note: In a real pipeline, we'd want a proper cell segmentation. 
        # Without it, we assume the nucleus mask is the primary object of interest.
        pass 

    # If mask is still empty or invalid, return 0
    if labeled_mask is None or np.max(labeled_mask) == 0:
        return 0.0

    # Calculate Polarity for each cell
    polarities = []
    
    # We use regionprops to get the centroid of the mask (Nucleus Center approximation)
    # and the intensity-weighted centroid of the Tubulin channel.
    
    # Note: If the mask provided is a Whole Cell mask, the geometric centroid might not be the nucleus center.
    # However, usually, the segmentation mask provided is either nuclei or cells.
    # If it's a cell mask, the geometric centroid is the cell center.
    # To be precise:
    # Reference Point: Geometric Centroid of the provided mask (assuming it centers on the cell/nucleus).
    # Signal Point: Weighted Centroid of Tubulin intensity within that mask.
    
    props = regionprops(labeled_mask, intensity_image=tubulin_norm)
    
    for prop in props:
        # Skip very small objects (noise)
        if prop.area < 50:
            continue
            
        # 1. Geometric Centroid of the mask (y, x)
        # This represents the center of the "container" (nucleus or cell body)
        y0, x0 = prop.centroid
        
        # 2. Intensity-weighted Centroid of Tubulin (y, x)
        # This represents where the tubulin mass is concentrated
        # If the region has 0 intensity sum, weighted_centroid returns the geometric centroid.
        yw, xw = prop.weighted_centroid
        
        # 3. Calculate Euclidean distance (Polarity magnitude)
        # Vector = (yw - y0, xw - x0)
        displacement = np.sqrt((yw - y0)**2 + (xw - x0)**2)
        
        polarities.append(displacement)

    # Aggregate results
    if not polarities:
        return 0.0
        
    # Return the mean polarity index across the population
    result = np.mean(polarities)

    return float(result)
