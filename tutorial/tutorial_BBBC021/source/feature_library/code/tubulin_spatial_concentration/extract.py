def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    
    # 1. Data Loading and Preprocessing
    # Convert to float32 for calculations
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensions (Expect 512x512x3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0
        
    # Extract Tubulin channel (Channel 1 - Green)
    tubulin = arr[..., 1]
    
    # Extract DAPI channel (Channel 2 - Blue) for fallback segmentation
    dapi = arr[..., 2]

    # Normalize Tubulin intensity to [0, 1]
    # Robust normalization using percentiles to handle outliers
    p_min, p_max = np.percentile(tubulin, (1, 99))
    if p_max > p_min:
        tubulin = (tubulin - p_min) / (p_max - p_min)
    else:
        tubulin = tubulin - p_min # Should be near 0
    tubulin = np.clip(tubulin, 0.0, 1.0)

    # 2. Segmentation Handling
    cell_labels = None
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use provided segmentation mask
        # Assuming the first mask is a cell mask or nuclei mask that identifies objects
        mask_input = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == tubulin.shape:
            # If mask is already labeled (int), use it. If binary, label it.
            if np.issubdtype(mask_input.dtype, np.integer) and mask_input.max() > 1:
                cell_labels = mask_input
            else:
                cell_labels = label(mask_input > 0)
    
    # Fallback Segmentation if no valid mask provided
    if cell_labels is None:
        # Simple nuclei detection on DAPI channel
        try:
            thresh = threshold_otsu(dapi)
            nuclei_mask = dapi > thresh
            # Label connected components
            cell_labels = label(nuclei_mask)
        except Exception:
            # If Otsu fails (e.g., empty image), return 0
            return 0.0

    # 3. Feature Calculation: Tubulin Spatial Concentration
    # Metric: Ratio of Intensity Radius of Gyration to Shape Radius of Gyration
    # Ratio < 1.0 implies concentration near centroid (e.g., mitotic spindle)
    # Ratio ~ 1.0 implies uniform distribution
    # Ratio > 1.0 implies cortical distribution
    
    props = regionprops(cell_labels, intensity_image=tubulin)
    
    concentration_scores = []
    
    for prop in props:
        # Skip very small artifacts
        if prop.area < 50:
            continue
            
        # Get coordinates of all pixels in the object
        # coords is (N, 2) array of (row, col)
        coords = prop.coords
        
        # Get intensities of these pixels
        intensities = prop.image_intensity[prop.image]  # Extract intensities within the bounding box mask
        
        # Calculate Centroid
        # We use the weighted centroid (intensity center of mass) if available, 
        # or geometric centroid. For "concentration relative to center", 
        # geometric centroid is usually the reference point to measure if intensity is centered.
        # However, biologically, microtubules radiate from the centrosome (near nucleus).
        # Using the geometric centroid of the cell/nucleus mask is a good proxy for the cell center.
        centroid = np.array(prop.centroid)
        
        # Calculate squared distances from centroid for every pixel
        # coords are absolute, centroid is absolute
        diffs = coords - centroid
        sq_distances = np.sum(diffs**2, axis=1) # (N,) array of r^2
        
        # 1. Radius of Gyration (Shape) - "Uniform distribution reference"
        # Rg_shape = sqrt( sum(1 * r^2) / sum(1) )
        # This measures how spread out the pixels are physically
        rg_shape = np.sqrt(np.mean(sq_distances))
        
        # 2. Radius of Gyration (Intensity) - "Actual distribution"
        # Rg_intensity = sqrt( sum(I * r^2) / sum(I) )
        total_intensity = np.sum(intensities)
        
        if total_intensity <= 0 or rg_shape <= 0:
            continue
            
        rg_intensity = np.sqrt(np.sum(intensities * sq_distances) / total_intensity)
        
        # Calculate Ratio: Rg_intensity / Rg_shape
        # Lower value = More concentrated near center
        ratio = rg_intensity / rg_shape
        concentration_scores.append(ratio)

    # 4. Aggregation
    if not concentration_scores:
        return 0.0
        
    # Return the median score across all cells in the image
    # We invert the logic slightly for the final return: 
    # The prompt asks for "concentration". 
    # The ratio calculated (Rg_int / Rg_shape) is a measure of *spread*.
    # Low ratio = High concentration.
    # To make the feature value intuitive (Higher = More Concentrated), we can return 1/ratio or (1 - ratio).
    # However, standard "Spatial Concentration" often refers to the raw second moment ratio or similar.
    # Let's stick to the raw ratio which is a standard morphological feature (often called "Radial Distribution").
    # But to align with "Concentration", a lower value is "more concentrated".
    # Let's return the median ratio directly. Users can interpret low values as high concentration.
    
    return float(np.median(concentration_scores))
