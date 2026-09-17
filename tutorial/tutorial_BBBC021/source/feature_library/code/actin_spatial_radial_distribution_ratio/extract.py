def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_erosion, disk

    # Convert to appropriate array type and normalize
    # Image is (512, 512, 3), uint8. Channel 0 is Actin.
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensionality
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Actin channel (Channel 0)
    actin_channel = arr[:, :, 0]

    # Normalize intensity to [0, 1]
    vmax = np.percentile(actin_channel, 99.5) if actin_channel.size > 0 else 1.0
    if vmax > 0:
        actin_channel = actin_channel / vmax
    actin_channel = np.clip(actin_channel, 0.0, 1.0)

    # Determine Segmentation Mask
    # We need a cell mask (cytoplasm + nucleus) to define the boundary.
    # If masks are provided, we try to use them. If not, we generate one from Actin.
    
    cell_mask = None
    
    # Check provided masks
    if len(segmentation_masks) > 0:
        # Heuristic: If multiple masks, usually the larger one covers the whole cell (cytoplasm)
        # or specific naming conventions might apply, but here we receive them as args.
        # We'll check for a mask that covers a significant portion of the image but not all.
        # Often: mask 0 is cells/cytoplasm, mask 1 is nuclei.
        # We will iterate and pick the one with the largest foreground area that isn't empty.
        best_mask = None
        max_area = 0
        for mask in segmentation_masks:
            if mask is None: continue
            # Ensure mask is 2D
            if mask.ndim == 3:
                mask = np.max(mask, axis=0) # Project if 3D
            if mask.shape != actin_channel.shape:
                continue # Skip mismatching shapes
                
            current_area = np.sum(mask > 0)
            if current_area > max_area:
                max_area = current_area
                best_mask = mask
        
        if best_mask is not None:
            cell_mask = best_mask

    # Fallback: Generate mask from Actin channel if no valid mask provided
    if cell_mask is None:
        # Simple background subtraction and thresholding
        # Gaussian blur to reduce noise
        blurred = ndimage.gaussian_filter(actin_channel, sigma=2)
        try:
            thresh = threshold_otsu(blurred)
            cell_mask = blurred > thresh
        except Exception:
            # Fallback for very low signal images
            cell_mask = blurred > 0.1

    # Ensure mask is labeled (instance segmentation)
    # If the mask is binary (0 and 1), label it. If it's already integer labels, keep it.
    if cell_mask.dtype == bool or np.max(cell_mask) == 1:
        labeled_mask = label(cell_mask)
    else:
        labeled_mask = cell_mask.astype(int)

    props = regionprops(labeled_mask)
    
    if not props:
        return 0.0

    ratios = []

    # Process each cell individually
    for prop in props:
        # Extract the bounding box for efficiency
        minr, minc, maxr, maxc = prop.bbox
        
        # Get local mask and intensity image
        # prop.image is the binary mask of the object in the bounding box
        local_mask = prop.image
        local_intensity = actin_channel[minr:maxr, minc:maxc]
        
        # Skip very small artifacts
        if local_mask.sum() < 50:
            continue

        # Compute Distance Transform on the local mask
        # dt gives distance to the nearest background pixel (0)
        # High values are deep inside the cell, low values (1) are at the edge.
        dt = ndimage.distance_transform_edt(local_mask)
        
        max_dist = np.max(dt)
        if max_dist == 0:
            continue

        # Define "Outer" (Cortex) vs "Inner" regions
        # Cortex: Outer 10% of the "radius" (distance to background)
        # Inner: The rest
        # Note: dt values are 1 at the edge, increasing inwards.
        # So "Outer" corresponds to low dt values.
        
        # Threshold for the boundary between cortex and inner
        # If max_dist is 20 pixels, the outer 10% is roughly pixels with distance <= 2
        # However, distance transform is 1-based for the first pixel inside.
        # Let's define the cortex as the rim.
        
        # Definition: Outer 10% of the radius.
        # Radius approx = max_dist.
        # Inner region starts at distance > 0.1 * max_dist?
        # Actually, usually "outer 10%" implies the rim.
        # Let's define the cut-off.
        
        # If the cell is very thin (max_dist < 3), the whole thing is effectively cortex.
        if max_dist <= 3:
            # Treat whole cell as cortex, ratio is effectively 1.0 (or undefined, but 1.0 implies uniform/cortical)
            # But to avoid skewing stats with tiny bits, we might skip or set to 1.
            ratios.append(1.0)
            continue
            
        # Define boundary
        boundary_dist = 0.2 * max_dist # Using 20% to ensure we capture a robust rim, 10% might be too thin for digital raster
        
        # Mask for outer region (Cortex): distance to background is small
        outer_mask = (dt <= boundary_dist) & local_mask
        
        # Mask for inner region: distance to background is large
        inner_mask = (dt > boundary_dist) & local_mask
        
        # If inner mask is empty (shouldn't happen if max_dist > 3 and boundary is 20%), handle it
        if np.sum(inner_mask) == 0:
            ratios.append(1.0)
            continue
            
        # Compute Mean Intensities
        # We use the local intensity slice masked by outer/inner masks
        mean_outer = np.mean(local_intensity[outer_mask])
        mean_inner = np.mean(local_intensity[inner_mask])
        
        # Compute Ratio
        # Add epsilon to avoid division by zero
        ratio = mean_outer / (mean_inner + 1e-7)
        ratios.append(ratio)

    if not ratios:
        return 0.0

    # Return the median ratio across all cells to represent the image phenotype
    # Median is robust to outliers (e.g., segmentation errors, debris)
    result = np.median(ratios)

    return float(result)
