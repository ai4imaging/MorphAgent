def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import binary_erosion, disk
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected shape: (512, 512, 3)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Channel 0 is Actin (Red)
        actin_channel = arr[:, :, 0]
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on desc, but safe)
        actin_channel = arr
    else:
        return 0.0

    # Intensity normalization
    # Background subtraction is crucial for ratio features to be meaningful
    # Estimate background from the lower percentile
    bg_val = np.percentile(actin_channel, 5)
    actin_channel = np.maximum(actin_channel - bg_val, 0)
    
    # Normalize to 0-1 range for stability, though ratio is scale-invariant
    vmax = np.percentile(actin_channel, 99.5)
    if vmax > 1e-6:
        actin_channel = actin_channel / vmax
    
    # Determine Cell Mask
    cell_mask = None
    
    # Check if segmentation masks are provided
    # We prioritize a cell mask (often the larger mask if multiple are present)
    if len(segmentation_masks) > 0:
        # Heuristic: Use the first mask provided. 
        # In many pipelines, mask[0] is cells/cytoplasm and mask[1] is nuclei.
        # We need the whole cell definition to define the cortex.
        candidate_mask = segmentation_masks[0]
        if candidate_mask.shape == actin_channel.shape:
            cell_mask = candidate_mask
    
    # Fallback: Generate mask if none provided
    if cell_mask is None:
        try:
            thresh = threshold_otsu(actin_channel)
            cell_mask = actin_channel > thresh
            # Clean up mask: remove small objects, fill holes
            cell_mask = ndimage.binary_fill_holes(cell_mask)
            # Label connected components
            labeled_mask = label(cell_mask)
        except Exception:
            return 0.0
    else:
        # Ensure mask is labeled (instance segmentation)
        if np.max(cell_mask) == 1 and np.unique(cell_mask).size <= 2:
            labeled_mask = label(cell_mask)
        else:
            labeled_mask = cell_mask.astype(int)

    # Feature Computation: Actin Cortical Ratio
    # Logic: For each cell, define Cortex (outer ring) and Center (inner region).
    # Ratio = Mean_Intensity(Cortex) / Mean_Intensity(Center)
    
    props = regionprops(labeled_mask, intensity_image=actin_channel)
    
    ratios = []
    
    # Erosion radius to define the cortex thickness.
    # For 512x512 MCF-7 cells, a 3-5 pixel ring is usually sufficient for the cortex.
    erosion_radius = 4
    selem = disk(erosion_radius)
    
    for prop in props:
        # Skip very small objects that can't support an erosion
        if prop.area < 100: 
            continue
            
        # Extract the local mask and intensity image for the single cell
        # regionprops provides a bounding box slice
        local_mask = prop.image  # Binary mask of the cell in the bbox
        local_intensity = prop.intensity_image
        
        # Define Center by eroding the full mask
        # We pad the local mask slightly to avoid border artifacts during erosion if the cell touches the bbox edge
        padded_mask = np.pad(local_mask, erosion_radius + 1, mode='constant', constant_values=0)
        eroded_padded = binary_erosion(padded_mask, selem)
        
        # Crop back to original bbox size
        local_center_mask = eroded_padded[erosion_radius+1:-erosion_radius-1, erosion_radius+1:-erosion_radius-1]
        
        # Ensure shapes match (handle potential off-by-one from padding/slicing edge cases)
        if local_center_mask.shape != local_mask.shape:
            continue

        # Define Cortex: Full Mask XOR Center Mask
        # Cortex is the pixels removed by erosion
        local_cortex_mask = np.logical_xor(local_mask, local_center_mask)
        
        # Check if we have valid regions
        if np.sum(local_center_mask) == 0 or np.sum(local_cortex_mask) == 0:
            continue
            
        # Compute Mean Intensities
        mean_cortex = np.mean(local_intensity[local_cortex_mask])
        mean_center = np.mean(local_intensity[local_center_mask])
        
        # Compute Ratio
        # Add epsilon to denominator to avoid division by zero
        ratio = mean_cortex / (mean_center + 1e-6)
        ratios.append(ratio)

    # Aggregate results
    if not ratios:
        return 0.0
        
    # Return the median ratio across the population
    # Median is robust to segmentation errors (e.g. debris)
    result = np.median(ratios)

    return float(result)
