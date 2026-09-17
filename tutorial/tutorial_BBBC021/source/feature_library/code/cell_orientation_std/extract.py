def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_closing, disk
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected 3-channel image, return 0.0
        return 0.0

    # Channel Selection
    # Channel 0 (Red/Actin) defines the cytoskeleton/cell shape best for orientation.
    # Channel 2 (Blue/DAPI) is useful for seeding segmentation if masks are missing.
    actin_channel = arr[..., 0]
    dapi_channel = arr[..., 2]

    # Intensity normalization for processing
    def normalize(image_channel):
        vmax = np.percentile(image_channel, 99.5) if image_channel.size > 0 else 1.0
        if vmax <= 0: vmax = 1.0
        return np.clip(image_channel / vmax, 0.0, 1.0)

    actin_norm = normalize(actin_channel)
    
    # Determine Segmentation Mask
    labeled_mask = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == arr.shape[:2]:
            # If mask is already labeled (int), use it. If boolean/binary, label it.
            if np.issubdtype(mask_input.dtype, np.integer) and mask_input.max() > 1:
                labeled_mask = mask_input
            else:
                labeled_mask = label(mask_input > 0)
    
    # Fallback: Generate segmentation if no valid mask provided
    if labeled_mask is None:
        # 1. Detect Nuclei (Seeds)
        dapi_norm = normalize(dapi_channel)
        thresh_val = threshold_otsu(dapi_norm) if dapi_norm.max() > dapi_norm.min() else 0.5
        nuclei_mask = dapi_norm > thresh_val
        nuclei_mask = binary_closing(nuclei_mask, disk(2))
        
        # Label nuclei markers
        markers = label(nuclei_mask)
        
        # 2. Detect Cell Boundaries (Watershed)
        # Use Actin channel gradient or intensity as the elevation map
        # We invert intensity because watershed fills basins (dark regions)
        elevation_map = -actin_norm 
        
        # Create a rough mask of where cells are (foreground) to limit watershed
        actin_thresh = threshold_otsu(actin_norm) if actin_norm.max() > actin_norm.min() else 0.1
        cell_mask = actin_norm > actin_thresh
        
        labeled_mask = watershed(elevation_map, markers, mask=cell_mask)

    # Feature Computation: Orientation Standard Deviation
    props = regionprops(labeled_mask)
    
    if len(props) < 2:
        return 0.0

    orientations = []
    
    for prop in props:
        # Filter out very small artifacts
        if prop.area < 50:
            continue
            
        # Filter out nearly circular objects
        # Orientation is unstable/undefined for circles (eccentricity ~ 0)
        # Eccentricity: 0 = circle, 1 = line.
        # We require some elongation to define a meaningful axis.
        if prop.eccentricity < 0.2:
            continue
            
        # prop.orientation is in range [-pi/2, pi/2]
        orientations.append(prop.orientation)

    if len(orientations) < 2:
        return 0.0

    orientations = np.array(orientations)

    # Circular Statistics Calculation
    # Cell orientation is axial data (headless vectors), with periodicity pi (180 degrees).
    # -89 degrees is close to +89 degrees.
    # To use standard circular statistics (which assume 2pi periodicity), we double the angles.
    # range becomes [-pi, pi].
    doubled_angles = 2 * orientations
    
    # Calculate Mean Resultant Vector Length (R)
    # R = sqrt( (sum(cos(angles)))^2 + (sum(sin(angles)))^2 ) / N
    # This avoids importing scipy.stats.circstd if we want to be minimal, 
    # but using the formula directly is safer and standard.
    C = np.sum(np.cos(doubled_angles))
    S = np.sum(np.sin(doubled_angles))
    R = np.sqrt(C**2 + S**2) / len(doubled_angles)
    
    # Circular Standard Deviation formula: sqrt( -2 * ln(R) )
    # Handle edge case where R is very close to 1 (variance ~ 0) or 0 (variance ~ inf)
    R = np.clip(R, 1e-9, 1.0)
    circ_std_doubled = np.sqrt(-2 * np.log(R))
    
    # Convert back to original scale (divide by 2)
    circ_std = circ_std_doubled / 2.0

    return float(circ_std)
