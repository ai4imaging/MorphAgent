def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu, threshold_triangle
    from skimage.measure import label, regionprops
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    # 1. Data Loading and Preprocessing
    # Convert to float32 for calculations
    img_arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality and extract channels
    # Expected shape: (H, W, 3) or (H, W) if single channel
    if img_arr.ndim == 3 and img_arr.shape[2] >= 3:
        # Channel 0 is Actin (Red), Channel 2 is Nuclei (Blue)
        actin_img = img_arr[..., 0]
        nuclei_img = img_arr[..., 2]
    elif img_arr.ndim == 2:
        # Fallback: treat the single channel as actin
        actin_img = img_arr
        nuclei_img = img_arr
    else:
        return 0.0

    # Normalize actin image to [0, 1] for intensity measurements
    # Robust normalization using percentiles
    p_min, p_max = np.percentile(actin_img, (1, 99))
    if p_max > p_min:
        actin_norm = (actin_img - p_min) / (p_max - p_min)
    else:
        actin_norm = actin_img
    actin_norm = np.clip(actin_norm, 0, 1)

    # 2. Segmentation Logic
    # We need a whole-cell segmentation mask.
    labeled_cells = None

    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Iterate through masks to find a likely cell mask (usually larger area than nuclei)
        # If multiple masks, we assume the one with larger average object size is the cell mask
        best_mask = None
        max_avg_area = 0
        
        for mask in segmentation_masks:
            if mask is None: continue
            mask_arr = np.asarray(mask).squeeze()
            if mask_arr.ndim != 2: continue
            
            # Simple check: is it labeled?
            if mask_arr.max() == 0: continue
            
            # Calculate average area to distinguish nuclei from cells
            # (Cells are generally larger than nuclei)
            props = regionprops(mask_arr.astype(int))
            if not props: continue
            avg_area = np.mean([p.area for p in props])
            
            if avg_area > max_avg_area:
                max_avg_area = avg_area
                best_mask = mask_arr
        
        if best_mask is not None:
            labeled_cells = best_mask.astype(int)

    # Fallback: Generate segmentation if no valid mask provided
    if labeled_cells is None:
        # Step A: Segment Nuclei (Seeds)
        try:
            thresh_nuc = threshold_otsu(nuclei_img)
            nuclei_mask = nuclei_img > thresh_nuc
            # Clean up nuclei
            nuclei_mask = ndimage.binary_opening(nuclei_mask, structure=np.ones((3,3)))
            markers = label(nuclei_mask)
        except:
            # If otsu fails (e.g. empty image), return 0
            return 0.0

        # Step B: Segment Cytoplasm (Mask)
        try:
            # Triangle is often good for actin which has a long tail in histogram
            thresh_actin = threshold_triangle(actin_img)
            cell_mask = actin_img > thresh_actin
            # Fill holes
            cell_mask = ndimage.binary_fill_holes(cell_mask)
        except:
            return 0.0

        # Step C: Watershed
        if np.max(markers) > 0:
            # Use gradient of actin as elevation map or just inverse intensity
            # Here we use inverse intensity so bright actin regions are basins? 
            # Actually, standard watershed on distance transform or gradient is common.
            # Let's use simple watershed: expand nuclei into cell mask
            labeled_cells = watershed(-actin_img, markers, mask=cell_mask)
        else:
            return 0.0

    # 3. Feature Extraction: Actin Radial Distribution Ratio
    # Ratio = (Mean Intensity of Outer 20%) / (Mean Intensity of Inner 50%)
    
    ratios = []
    
    # Get properties of each cell
    props = regionprops(labeled_cells, intensity_image=actin_norm)
    
    for prop in props:
        # Skip small artifacts
        if prop.area < 100:
            continue
            
        # Extract the bounding box image for efficiency
        # prop.image is the binary mask of the cell in the bbox
        # prop.intensity_image is the intensity image in the bbox
        cell_mask_local = prop.image
        cell_intensity_local = prop.intensity_image
        
        # Compute Distance Transform on the local mask
        # edt calculates distance to the nearest background pixel (0)
        # So: 0 at boundary, increasing towards center
        dist_map = ndimage.distance_transform_edt(cell_mask_local)
        
        max_dist = np.max(dist_map)
        if max_dist <= 1.0:
            continue # Too thin to measure radial distribution
            
        # Define Regions
        # Outer 20%: Distance is small (close to boundary)
        # Inner 50%: Distance is large (close to center)
        # Note: "Outer 20%" usually means the outer 20% of the radius.
        # If R is max_dist, Outer is [0, 0.2*R], Inner is [0.5*R, R]
        
        outer_threshold = 0.2 * max_dist
        inner_threshold = 0.5 * max_dist
        
        # Create masks for regions
        # dist_map > 0 ensures we are inside the cell
        outer_region_mask = (dist_map > 0) & (dist_map <= outer_threshold)
        inner_region_mask = dist_map >= inner_threshold
        
        # Calculate Mean Intensities
        # Use epsilon to avoid division by zero
        mean_outer = 0.0
        mean_inner = 0.0
        
        pixels_outer = cell_intensity_local[outer_region_mask]
        if pixels_outer.size > 0:
            mean_outer = np.mean(pixels_outer)
            
        pixels_inner = cell_intensity_local[inner_region_mask]
        if pixels_inner.size > 0:
            mean_inner = np.mean(pixels_inner)
        
        # Compute Ratio
        # If inner is 0 (e.g. hollow cell or artifact), handle gracefully
        if mean_inner < 1e-6:
            # If outer is significant, ratio is high. Cap at a reasonable number.
            if mean_outer > 0.05: # arbitrary noise floor
                ratio = 5.0 # High ratio indicating cortical enrichment
            else:
                ratio = 1.0 # Both dark
        else:
            ratio = mean_outer / mean_inner
            
        ratios.append(ratio)

    # 4. Aggregation
    if not ratios:
        return 0.0
        
    # Use median to be robust against segmentation errors
    return float(np.median(ratios))
