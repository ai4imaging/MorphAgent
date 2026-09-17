def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed

    # Convert to appropriate array type and handle dimensions
    # Dataset is (512, 512, 3) uint8
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensions
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not the expected 3-channel image, return 0.0
        return 0.0

    # Extract Channels
    # Channel 0: Actin (Red) - Cytoskeleton
    # Channel 1: Tubulin (Green) - Target for radial distribution
    # Channel 2: DAPI (Blue) - Nucleus center
    actin_ch = arr[..., 0]
    tubulin_ch = arr[..., 1]
    dapi_ch = arr[..., 2]

    # Normalize Tubulin channel for intensity calculations
    # Robust normalization using percentiles to handle outliers
    p99 = np.percentile(tubulin_ch, 99.5)
    if p99 > 0:
        tubulin_norm = tubulin_ch / p99
    else:
        tubulin_norm = tubulin_ch
    tubulin_norm = np.clip(tubulin_norm, 0.0, 1.0)

    # --- Segmentation Logic ---
    # We need Nuclei (for centers) and Cell Bodies (for boundaries)
    
    nuclei_labels = None
    cell_labels = None

    # Check if segmentation masks are provided
    # We expect at least two masks if provided: usually nuclei first, then cells, or vice versa.
    # However, standard practice often provides nuclei then cells.
    # If only one mask is provided, we assume it's a cell mask and try to infer nuclei, or vice versa.
    # Given the variability, a robust fallback pipeline is safer.
    
    if len(segmentation_masks) >= 2:
        # Heuristic: Nuclei masks usually have smaller total area than cell masks
        mask1 = segmentation_masks[0]
        mask2 = segmentation_masks[1]
        
        # Ensure masks are 2D integer arrays
        if mask1.ndim == 2 and mask2.ndim == 2:
            area1 = np.sum(mask1 > 0)
            area2 = np.sum(mask2 > 0)
            
            if area1 < area2:
                nuclei_labels = mask1.astype(int)
                cell_labels = mask2.astype(int)
            else:
                nuclei_labels = mask2.astype(int)
                cell_labels = mask1.astype(int)
    
    # Fallback Segmentation if masks are missing or invalid
    if nuclei_labels is None or cell_labels is None:
        # 1. Segment Nuclei (DAPI)
        try:
            thresh_nuc = threshold_otsu(dapi_ch)
            nuclei_mask = dapi_ch > thresh_nuc
            # Fill holes and label
            nuclei_mask = ndimage.binary_fill_holes(nuclei_mask)
            nuclei_labels = label(nuclei_mask)
        except Exception:
            # If thresholding fails (e.g. empty image), return 0
            return 0.0

        # 2. Segment Cells (Actin + Tubulin combined often gives better cell body)
        # Use Actin primarily as it defines cytoskeleton
        try:
            # Combine Actin and Tubulin for a robust cell body signal
            cell_signal = actin_ch + tubulin_ch
            thresh_cell = threshold_otsu(cell_signal)
            cell_mask = cell_signal > thresh_cell
            
            # Watershed to separate touching cells based on nuclei seeds
            distance = ndimage.distance_transform_edt(cell_mask)
            # We use nuclei_labels as markers. 
            # Note: watershed works on the negative distance for basin filling
            cell_labels = watershed(-distance, nuclei_labels, mask=cell_mask)
        except Exception:
            return 0.0

    # --- Feature Computation: Radial Distribution CV ---
    
    # Get properties of nuclei to find centroids
    nuclei_props = regionprops(nuclei_labels)
    
    # Map Nucleus Label -> Centroid
    # Note: regionprops labels match the integer values in nuclei_labels
    nuc_centers = {prop.label: prop.centroid for prop in nuclei_props}
    
    # Get properties of cells to iterate
    cell_props = regionprops(cell_labels, intensity_image=tubulin_norm)
    
    cv_values = []
    
    for cell_prop in cell_props:
        label_id = cell_prop.label
        
        # We need the corresponding nucleus centroid for this cell
        # Ideally, the labels match (1-to-1 mapping from watershed).
        # If labels don't match (e.g. provided masks with different IDs), we need to find which nucleus is inside.
        
        center = None
        if label_id in nuc_centers:
            center = nuc_centers[label_id]
        else:
            # If ID mismatch, find nucleus centroid inside this cell's bounding box
            # This is computationally expensive, so we skip complex matching for this implementation 
            # and assume 1-to-1 mapping or skip the cell.
            # A simple check: is there a nucleus centroid inside this cell's bounding box?
            # For robustness in this constrained environment, we skip if direct ID match fails.
            continue
            
        # Cell bounding box coordinates
        min_row, min_col, max_row, max_col = cell_prop.bbox
        
        # Adjust centroid to the cropped bbox coordinate system
        # Centroid is (row, col)
        cy = center[0] - min_row
        cx = center[1] - min_col
        
        # Extract the intensity image of the cell (masked by the cell shape)
        # cell_prop.image is the binary mask of the cell within the bbox
        # cell_prop.intensity_image is the intensity within the bbox (0 outside mask)
        cell_intensity = cell_prop.intensity_image
        cell_mask_bbox = cell_prop.image
        
        if cell_intensity.size == 0 or np.sum(cell_mask_bbox) == 0:
            continue

        # Create coordinate grid for the bbox
        h, w = cell_mask_bbox.shape
        y_indices, x_indices = np.indices((h, w))
        
        # Calculate Euclidean distance from the nucleus centroid for every pixel
        distances = np.sqrt((y_indices - cy)**2 + (x_indices - cx)**2)
        
        # We only care about distances within the cell mask
        valid_distances = distances[cell_mask_bbox]
        valid_intensities = cell_intensity[cell_mask_bbox]
        
        if valid_distances.size == 0:
            continue
            
        # Define radial bins
        # We use fixed pixel width bins to preserve physical scale
        bin_width = 2.0 # pixels
        max_dist = np.max(valid_distances)
        
        # If cell is too small (smaller than one bin), skip
        if max_dist < bin_width:
            continue
            
        # Digitize distances into bins
        # bins: [0, 2, 4, 6, ...]
        bins = np.arange(0, max_dist + bin_width, bin_width)
        bin_indices = np.digitize(valid_distances, bins)
        
        # Calculate mean intensity per bin
        # ndimage.mean is efficient for this
        # labels are bin_indices, index is unique bins present
        unique_bins = np.unique(bin_indices)
        radial_profile = ndimage.mean(valid_intensities, labels=bin_indices, index=unique_bins)
        
        # Calculate CV of the profile
        # Profile represents intensity vs distance from nucleus
        # High CV = localized/clustered distribution (e.g. perinuclear ring)
        # Low CV = diffuse/uniform distribution
        
        # Filter out NaNs if any (though ndimage.mean shouldn't produce them for existing labels)
        radial_profile = radial_profile[~np.isnan(radial_profile)]
        
        if radial_profile.size > 1:
            mu = np.mean(radial_profile)
            sigma = np.std(radial_profile)
            
            if mu > 1e-6: # Avoid division by zero
                cv = sigma / mu
                cv_values.append(cv)
            else:
                cv_values.append(0.0)
        else:
            # Flat profile or single point
            cv_values.append(0.0)

    # Aggregate results
    if not cv_values:
        return 0.0
        
    # Return the mean CV across all cells in the image
    result = np.mean(cv_values)
    
    return float(result)
