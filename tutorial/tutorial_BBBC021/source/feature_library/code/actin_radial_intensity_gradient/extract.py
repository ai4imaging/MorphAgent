def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage, stats
    from skimage.filters import threshold_otsu
    from skimage.measure import label, regionprops
    
    # Convert to appropriate array type and normalize
    arr = np.asarray(img, dtype=np.float32)
    
    # Handle dimensionality and extract Actin channel (Channel 0)
    # Dataset is (512, 512, 3) RGB TIFF. Channel 0 = Actin.
    if arr.ndim == 3 and arr.shape[2] == 3:
        actin_img = arr[:, :, 0]
    elif arr.ndim == 2:
        # Fallback if single channel passed, though unlikely given dataset description
        actin_img = arr
    else:
        return 0.0

    # Intensity normalization [0, 1]
    # Robust max to avoid hot pixels skewing the range
    vmax = np.percentile(actin_img, 99.5) if actin_img.size > 0 else 1.0
    if vmax > 0:
        actin_img = actin_img / vmax
    actin_img = np.clip(actin_img, 0.0, 1.0)

    # Handle Segmentation
    # We need a labeled mask of cells/cytoplasm to compute radial gradients per cell.
    labeled_mask = None
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask. Assuming it's a cell/cytoplasm mask.
        # Ensure it matches image dimensions (handle potential 3D vs 2D mismatch if any)
        mask_input = segmentation_masks[0]
        if mask_input.shape == actin_img.shape:
            labeled_mask = mask_input.astype(int)
    
    # Fallback: Generate mask if none provided
    if labeled_mask is None:
        # Simple background subtraction and thresholding
        # Blur to reduce noise
        blurred = ndimage.gaussian_filter(actin_img, sigma=2)
        try:
            thresh = threshold_otsu(blurred)
            binary_mask = blurred > thresh
            # Label connected components
            labeled_mask = label(binary_mask)
        except Exception:
            # If image is empty or uniform, otsu might fail
            return 0.0

    # Feature Computation: Actin Radial Intensity Gradient
    # Logic: For each cell, compute the correlation between radial distance from center
    # and actin intensity.
    # Positive gradient -> Brighter at edges (Cortical ring)
    # Negative/Zero gradient -> Brighter at center or uniform (Diffuse/Stress fibers)

    props = regionprops(labeled_mask, intensity_image=actin_img)
    
    slopes = []

    for prop in props:
        # Skip very small artifacts
        if prop.area < 100:
            continue

        # Extract the bounding box of the cell to reduce computation
        minr, minc, maxr, maxc = prop.bbox
        
        # Get the binary mask for just this cell within the bbox
        cell_mask_local = prop.image  # Binary mask of the object in bbox
        
        # Get the intensity image for this cell
        cell_intensity_local = prop.intensity_image
        
        # Compute Distance Transform
        # edt calculates distance to the nearest background (0) pixel.
        # The "center" of the cell has the highest value.
        # The "edge" of the cell has the lowest value (1).
        dist_map = ndimage.distance_transform_edt(cell_mask_local)
        
        max_dist = dist_map.max()
        if max_dist <= 1.0:
            continue # Too thin to calculate a gradient

        # Normalize distance: 
        # We want 0.0 at the center (deepest point) and 1.0 at the periphery.
        # dist_map / max_dist gives 1.0 at center, 0.0 at edge.
        # So we invert it: 1.0 - (dist / max)
        normalized_radial_dist = 1.0 - (dist_map / max_dist)
        
        # Flatten arrays using the mask to select only cell pixels
        # We only care about pixels inside the cell
        x = normalized_radial_dist[cell_mask_local]
        y = cell_intensity_local[cell_mask_local]
        
        if len(x) < 10: # Need enough points for regression
            continue
            
        # Compute linear regression slope (Gradient)
        # We want to know how Intensity (Y) changes as we move to Periphery (X increases)
        # Slope = Covariance(x, y) / Variance(x)
        # Using numpy for speed over scipy.stats.linregress
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            slope = 0.0
        else:
            slope = numerator / denominator
            
        slopes.append(slope)

    # Aggregate results
    # We return the median slope across the population to represent the dominant phenotype
    if not slopes:
        return 0.0
    
    result = np.median(slopes)
    
    return float(result)
