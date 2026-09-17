def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from scipy import stats

    # 1. Data Loading and Preprocessing
    # Ensure image is float32 for calculations
    img_arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality
    if img_arr.ndim != 3:
        return 0.0
    
    h, w, c = img_arr.shape
    if c != 3:
        # If channels are not last or not 3, try to adapt or fail gracefully
        if img_arr.shape[0] == 3: # Channel first
            img_arr = np.transpose(img_arr, (1, 2, 0))
        else:
            return 0.0

    # Extract relevant channels based on dataset description
    # Channel 1: Tubulin (Green) - The signal to measure
    # Channel 2: DAPI (Blue) - The center reference (Nuclei)
    tubulin = img_arr[:, :, 1]
    dapi = img_arr[:, :, 2]

    # Normalize intensities to [0, 1] robustly
    def normalize_channel(ch):
        p_max = np.percentile(ch, 99.5)
        if p_max > 0:
            return np.clip(ch / p_max, 0.0, 1.0)
        return np.zeros_like(ch)

    tubulin_norm = normalize_channel(tubulin)
    dapi_norm = normalize_channel(dapi)

    # 2. Define Cell Centers and Boundaries
    # We need to identify individual cells to calculate radial distributions per cell.
    # If segmentation masks are provided, use them. Otherwise, generate a fallback.
    
    nuclei_labels = None
    cell_labels = None
    
    # Check if valid segmentation masks are provided
    # Expectation: masks are passed as args. 
    # Common convention: (cell_mask, nuclei_mask) or just (nuclei_mask) depending on pipeline.
    # We will prioritize finding a nuclei mask to define centers.
    
    has_masks = len(segmentation_masks) > 0 and all(m is not None for m in segmentation_masks)
    
    if has_masks:
        # Heuristic: usually the smaller objects are nuclei, larger are cells.
        # Or based on order. Let's assume if 2 masks, [0] is cells, [1] is nuclei (or vice versa).
        # Without metadata, we rely on the DAPI channel to verify which mask overlaps best with nuclei.
        
        # Simple fallback logic for using provided masks:
        # Use the first mask as the "object" definition.
        # If multiple, we try to distinguish.
        # For this specific feature, we need centroids.
        
        # Let's try to use the first mask as the cell definition.
        # We still need centroids. We can compute centroids of the mask.
        ref_mask = segmentation_masks[0]
        if ref_mask.shape == (h, w):
            cell_labels = ref_mask.astype(np.int32)
            # If we have a second mask, maybe it's nuclei
            if len(segmentation_masks) > 1:
                nuclei_labels = segmentation_masks[1].astype(np.int32)
    
    # Fallback: Generate masks if not provided or invalid
    if cell_labels is None:
        # 1. Detect Nuclei (Otsu-like thresholding)
        # Simple threshold: mean + std
        thresh_val = np.mean(dapi_norm) + np.std(dapi_norm)
        nuclei_mask = dapi_norm > thresh_val
        
        # Clean up noise
        nuclei_mask = ndimage.binary_opening(nuclei_mask, structure=np.ones((3,3)))
        nuclei_labels, num_nuclei = ndimage.label(nuclei_mask)
        
        # 2. Define Cell Boundaries (Voronoi-like via Distance Transform Watershed)
        # Since we can't import watershed from skimage, we use a distance-based approximation
        # or simply assign pixels to the nearest nucleus label.
        if num_nuclei > 0:
            # Create a grid of coordinates
            y_indices, x_indices = np.indices((h, w))
            
            # Get centroids of nuclei
            centers = ndimage.center_of_mass(dapi_norm, nuclei_labels, range(1, num_nuclei + 1))
            centers = np.array(centers) # shape (N, 2)
            
            # If too many nuclei (e.g. noise), filter or limit
            if len(centers) > 200: # Safety cap for performance
                 # Just take the largest ones
                 sizes = ndimage.sum(nuclei_mask, nuclei_labels, range(1, num_nuclei + 1))
                 valid_indices = np.argsort(sizes)[::-1][:200]
                 centers = centers[valid_indices]
                 num_nuclei = len(centers)

            # Create an approximate cell mask using nearest neighbor (Voronoi)
            # Only consider pixels with some tubulin signal (foreground)
            tubulin_thresh = np.mean(tubulin_norm) # Background cut-off
            foreground_mask = tubulin_norm > tubulin_thresh
            
            if num_nuclei > 0 and np.any(foreground_mask):
                # Efficient nearest neighbor assignment for foreground pixels
                # We use scipy.ndimage.distance_transform_edt with return_indices=True
                # to find the nearest nucleus for every pixel
                
                # Create a marker map
                marker_map = np.zeros((h, w), dtype=np.int32)
                # Place markers at centroids (approximate)
                for idx, (cy, cx) in enumerate(centers):
                    r, c_idx = int(cy), int(cx)
                    if 0 <= r < h and 0 <= c_idx < w:
                        marker_map[r, c_idx] = idx + 1
                
                # If markers are sparse points, EDT can propagate labels
                # Invert marker map for EDT (0 is background for EDT)
                # We want distance to nearest non-zero marker
                # Actually, ndimage.distance_transform_edt computes distance to zero. 
                # So we need feature transform.
                # A simpler approach without complex watershed:
                # Just iterate cells.
                cell_labels = None # We will handle per-cell logic via centroids directly
            else:
                return 0.0
        else:
            return 0.0

    # 3. Compute Radial Distribution Gradient
    # The goal: Measure slope of intensity vs distance from nucleus center.
    # Monoastral = High intensity at center, rapid decay (Negative slope).
    # Diffuse = Flat slope.
    
    gradients = []
    
    # Get centroids again to be sure (either from fallback or provided masks)
    if nuclei_labels is not None:
        # Use provided nuclei labels
        objs = ndimage.find_objects(nuclei_labels)
        num_labels = len(objs)
        centers = ndimage.center_of_mass(dapi_norm, nuclei_labels, range(1, num_labels + 1))
        # Filter out NaNs
        centers = [c for c in centers if not np.any(np.isnan(c))]
    elif cell_labels is not None:
         # Use cell centers if no nuclei mask
         objs = ndimage.find_objects(cell_labels)
         num_labels = len(objs)
         centers = ndimage.center_of_mass(dapi_norm, cell_labels, range(1, num_labels + 1))
         centers = [c for c in centers if not np.any(np.isnan(c))]
    else:
        # Fallback centroids calculated earlier
        # If we didn't make labels but have centers from fallback logic
        if 'centers' not in locals() or len(centers) == 0:
            return 0.0
            
    # Parameters for radial profiling
    max_radius = 60  # pixels (approximate radius of a cell in 512x512)
    
    y_grid, x_grid = np.indices((h, w))
    
    for (cy, cx) in centers:
        # Define a Region of Interest (ROI) around the centroid to speed up
        r_int, c_int = int(cy), int(cx)
        y_min, y_max = max(0, r_int - max_radius), min(h, r_int + max_radius)
        x_min, x_max = max(0, c_int - max_radius), min(w, c_int + max_radius)
        
        roi_tubulin = tubulin_norm[y_min:y_max, x_min:x_max]
        roi_y = y_grid[y_min:y_max, x_min:x_max]
        roi_x = x_grid[y_min:y_max, x_min:x_max]
        
        # Calculate distances from center
        distances = np.sqrt((roi_y - cy)**2 + (roi_x - cx)**2)
        
        # Mask for valid radius and foreground
        # We only care about the gradient within the cell radius
        mask = (distances <= max_radius) & (distances > 2) # Exclude immediate center (often nucleus hole)
        
        if np.sum(mask) < 50: # Not enough pixels
            continue
            
        valid_dists = distances[mask]
        valid_intensities = roi_tubulin[mask]
        
        # Calculate Gradient (Slope)
        # We want the slope of Intensity = m * Distance + c
        # A monoastral cell has high intensity at low distance, low intensity at high distance -> Negative Slope
        # A diffuse cell has flat slope.
        
        # Linear regression
        # slope = cov(x, y) / var(x)
        if np.std(valid_dists) == 0:
            continue
            
        slope, _, _, _, _ = stats.linregress(valid_dists, valid_intensities)
        
        # We store the negative of the slope so that "Higher Value" = "Steeper Decay" = "More Monoastral"
        # Normal slope is negative for decay. -(-0.5) = 0.5.
        # However, the prompt asks for "gradient". Usually gradients are vectors, but here it implies magnitude/steepness.
        # Let's return the raw slope. A very negative slope indicates the phenotype.
        # But to make it a positive feature magnitude (often preferred for "strength" of phenotype):
        # Let's return the absolute decay.
        # Actually, let's stick to the raw slope. Monoastral = very negative. Control = near zero.
        # Wait, feature description: "Measures the gradient... designed to detect monoastral phenotype".
        # Usually features are constructed such that higher = more phenotype.
        # Monoastral = High Center -> Low Edge. Slope is negative.
        # Let's return the negative slope (decay rate). Positive value means intensity drops as we go out.
        gradients.append(-slope)

    if not gradients:
        return 0.0
        
    # Return the median decay rate across the population
    # Higher positive value = Stronger radial distribution (Monoastral)
    result = np.median(gradients)
    
    return float(result)
