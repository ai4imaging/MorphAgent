def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from skimage.morphology import disk, dilation

    # 1. Data Loading and Validation
    # Convert to float32 for processing
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensions: Expecting (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # 2. Channel Extraction
    # Channel 1: Tubulin (Green) - Target signal
    # Channel 2: DAPI (Blue) - Nuclei reference
    tubulin = arr[..., 1]
    dapi = arr[..., 2]

    # 3. Normalization
    # Normalize each channel to [0, 1] based on robust max
    def normalize(ch):
        vmax = np.percentile(ch, 99.5)
        if vmax <= 0: return ch
        return np.clip(ch / vmax, 0, 1)

    tubulin_norm = normalize(tubulin)
    dapi_norm = normalize(dapi)

    # 4. Segmentation Logic
    # We need to identify cells and their corresponding nuclei to measure radial distribution.
    # If segmentation masks are provided, use them. Otherwise, generate them.
    
    nuclei_mask = None
    cell_mask = None

    if len(segmentation_masks) > 0:
        # Assuming standard mask order if available, or just taking the first one
        # Often masks are passed as (nuclei, cells) or just (nuclei)
        # We will try to infer or just use the first valid one as nuclei
        # Based on typical pipelines, let's look for a mask that aligns with DAPI
        
        # Simple heuristic: if we have 2 masks, assume [0] is nuclei, [1] is cells (or vice versa, but usually nuclei is primary)
        # If we have 1 mask, assume it's nuclei/cells combined or just nuclei.
        
        # Let's try to generate our own robust segmentation to be safe, 
        # as the prompt implies we might need to handle the "no mask" case robustly 
        # and previous feedback suggested segmentation issues.
        # However, if masks are present, we should use them to respect the function signature intent.
        
        # Let's check the first mask.
        mask0 = segmentation_masks[0]
        if mask0.shape == dapi.shape:
            nuclei_mask = mask0
            
        # If a second mask exists, it might be the cell body
        if len(segmentation_masks) > 1:
            mask1 = segmentation_masks[1]
            if mask1.shape == dapi.shape:
                cell_mask = mask1
    
    # Fallback / Refinement Segmentation
    if nuclei_mask is None:
        # Otsu thresholding for nuclei
        # Simple manual threshold often works better for DAPI than Otsu on noisy images
        thresh_val = np.percentile(dapi_norm, 90) * 0.5 # Heuristic
        if thresh_val < 0.05: thresh_val = 0.05
        
        binary_dapi = dapi_norm > thresh_val
        # Remove small noise
        binary_dapi = ndimage.binary_opening(binary_dapi, structure=np.ones((3,3)))
        
        # Label nuclei
        nuclei_mask = label(binary_dapi)

    # If we don't have a cell mask, we need to estimate cell boundaries to define the "radius"
    # We use a watershed on the tubulin signal seeded by nuclei
    if cell_mask is None:
        # Smooth tubulin to get general cell shape
        tubulin_smooth = ndimage.gaussian_filter(tubulin_norm, sigma=2)
        # Threshold to get background vs cell
        t_thresh = np.percentile(tubulin_smooth, 20) # Low percentile for background
        if t_thresh < 0.02: t_thresh = 0.02
        mask_bg = tubulin_smooth > t_thresh
        
        # Watershed
        # We need markers. Use nuclei centroids or the nuclei mask itself.
        markers = nuclei_mask
        
        # Gradient of image is often used for watershed, or just inverted intensity
        # Here we just want to expand from nuclei to the edge of the signal
        cell_mask = watershed(-tubulin_smooth, markers, mask=mask_bg)

    # 5. Feature Calculation: Tubulin Radial Distribution
    # Metric: Weighted Mean Distance of Tubulin from Nucleus Centroid / Mean Radius of Cell
    # Or simpler: Fraction of intensity in outer shell vs inner shell.
    
    # Approach: For each cell, calculate the "Perinuclear Ratio".
    # High value = Tubulin concentrated near nucleus.
    # Low value = Tubulin spread to periphery.
    # The prompt asks for "tubulin_radial_distribution" which usually implies 
    # quantifying extension to periphery.
    # Let's define the feature as: Normalized Radial Moment.
    # Value = (Intensity-weighted mean distance from center) / (Max distance from center)
    # Higher value -> More peripheral. Lower value -> More perinuclear.
    
    props_nuclei = regionprops(nuclei_mask)
    
    # Map nuclei labels to cell labels (they should be same if watershed used nuclei as markers)
    # If masks came from outside, they might not match perfectly.
    # We will iterate through unique labels in the cell_mask.
    
    cell_labels = np.unique(cell_mask)
    cell_labels = cell_labels[cell_labels > 0]
    
    radial_scores = []
    
    for lbl in cell_labels:
        # Create a mask for the current cell
        current_cell_mask = (cell_mask == lbl)
        
        # Find the corresponding nucleus centroid
        # We look for the nucleus label that overlaps most with this cell
        # If generated via watershed, label 'lbl' in nuclei_mask corresponds to 'lbl' in cell_mask
        
        # Get coordinates of the cell pixels
        coords = np.argwhere(current_cell_mask)
        if coords.shape[0] < 50: # Ignore tiny fragments
            continue
            
        # Extract tubulin intensity for this cell
        cell_intensities = tubulin_norm[current_cell_mask]
        
        # Calculate centroid of the cell (or preferably the nucleus)
        # Let's try to find the nucleus center for this cell label
        # If nuclei_mask has the same label:
        if lbl <= nuclei_mask.max() and np.any(nuclei_mask == lbl):
            # Use specific nucleus centroid
            n_coords = np.argwhere(nuclei_mask == lbl)
            if n_coords.shape[0] > 0:
                centroid = np.mean(n_coords, axis=0)
            else:
                centroid = np.mean(coords, axis=0)
        else:
            # Fallback to cell centroid (less ideal for radial distribution relative to nucleus)
            centroid = np.mean(coords, axis=0)
            
        # Calculate distances of all cell pixels from the centroid
        # coords is (N, 2), centroid is (2,)
        distances = np.sqrt(np.sum((coords - centroid)**2, axis=1))
        
        # Avoid division by zero
        max_dist = np.max(distances)
        if max_dist == 0:
            continue
            
        # Calculate Intensity Weighted Mean Distance
        # Sum(Intensity * Distance) / Sum(Intensity)
        total_intensity = np.sum(cell_intensities)
        if total_intensity == 0:
            continue
            
        weighted_mean_dist = np.sum(cell_intensities * distances) / total_intensity
        
        # Normalize by the maximum radius of the cell to make it scale-invariant
        # Score ranges roughly [0, 1]
        # 0.0 -> All intensity at center
        # 1.0 -> All intensity at very edge
        # Uniform disk -> 2/3 (approx 0.66)
        normalized_radial_dist = weighted_mean_dist / max_dist
        
        radial_scores.append(normalized_radial_dist)

    # 6. Aggregation
    if not radial_scores:
        return 0.0
        
    # Return the mean score across all cells
    result = np.mean(radial_scores)
    
    return float(result)
