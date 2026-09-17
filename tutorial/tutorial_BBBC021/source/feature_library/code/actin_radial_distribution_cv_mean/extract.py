def extract(img, *segmentation_masks):
    import numpy as np
    from scipy import ndimage
    from skimage.measure import regionprops

    # 1. Prepare Image Data
    # Dataset is (512, 512, 3), Channel 0 is Actin (Red)
    # Convert to float32 and normalize to [0, 1]
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 3 and img.shape[2] >= 1:
        actin_channel = img[:, :, 0] / 255.0
    elif img.ndim == 2:
        actin_channel = img / 255.0
    else:
        return 0.0

    # 2. Handle Segmentation Masks
    # We need at least a cell mask to define boundaries.
    # Ideally, we have a nucleus mask to define the center.
    cell_mask = None
    nuclei_mask = None

    if len(segmentation_masks) > 0:
        # Heuristic: usually the larger mask is cytoplasm/cell, smaller is nucleus
        # Or based on order. Let's assume:
        # If 1 mask: it's the cell mask.
        # If 2 masks: check overlap or size to distinguish.
        
        mask1 = segmentation_masks[0]
        if len(segmentation_masks) >= 2:
            mask2 = segmentation_masks[1]
            # Simple heuristic: Nuclei are usually smaller and contained within cells
            sum1 = np.sum(mask1 > 0)
            sum2 = np.sum(mask2 > 0)
            if sum1 > sum2:
                cell_mask = mask1
                nuclei_mask = mask2
            else:
                cell_mask = mask2
                nuclei_mask = mask1
        else:
            cell_mask = mask1
            nuclei_mask = None # Will fallback to cell centroid
    
    # If no masks provided, we cannot compute per-cell radial distribution reliably
    # Fallback: Treat the whole image as one "cell" or return 0.0?
    # Given the specificity of the feature (radial distribution from center), 
    # without objects, it's ill-defined. Returning 0.0 is safer than noise.
    if cell_mask is None:
        return 0.0

    # Ensure masks are integer labeled
    cell_mask = cell_mask.astype(int)
    if nuclei_mask is not None:
        nuclei_mask = nuclei_mask.astype(int)

    # 3. Compute Feature Per Cell
    cell_props = regionprops(cell_mask)
    
    # Pre-calculate coordinate grids for distance computation
    h, w = actin_channel.shape
    y_indices, x_indices = np.indices((h, w))
    
    cv_values = []
    
    # Number of bins for the radial profile (e.g., 20 rings from center to edge)
    num_bins = 20

    for prop in cell_props:
        cell_id = prop.label
        
        # Get the bounding box to reduce computation area
        minr, minc, maxr, maxc = prop.bbox
        
        # Extract local mask and intensity
        local_cell_mask = (cell_mask[minr:maxr, minc:maxc] == cell_id)
        local_intensity = actin_channel[minr:maxr, minc:maxc]
        
        # Skip if cell is too small
        if np.sum(local_cell_mask) < 10:
            continue

        # Determine Center (Nucleus Centroid or Cell Centroid)
        center_y, center_x = prop.centroid # Default to cell centroid (relative to image)
        
        if nuclei_mask is not None:
            # Find the nucleus corresponding to this cell
            # We look for the most frequent label in the nuclei mask under the cell mask area
            # Or simply assume label consistency if labels match (often true in pipelines)
            # Let's try to find the nucleus centroid if it exists
            
            # Crop nuclei mask to same bbox
            local_nuc_mask = nuclei_mask[minr:maxr, minc:maxc]
            # Filter by cell mask to ensure we look inside the cell
            local_nuc_ids = local_nuc_mask[local_cell_mask]
            local_nuc_ids = local_nuc_ids[local_nuc_ids > 0]
            
            if local_nuc_ids.size > 0:
                # Find dominant nucleus ID
                nuc_id = np.bincount(local_nuc_ids).argmax()
                
                # Get centroid of that nucleus
                # We need to find it in the global or local scope
                # Efficient way: calculate center of mass of that specific ID in local crop
                ny, nx = ndimage.center_of_mass(local_nuc_mask == nuc_id)
                if not np.isnan(ny):
                    center_y = minr + ny
                    center_x = minc + nx

        # Adjust center to local bbox coordinates
        local_cy = center_y - minr
        local_cx = center_x - minc
        
        # Get coordinates of pixels belonging to the cell
        # We use the local grid
        ly, lx = np.indices((maxr-minr, maxc-minc))
        
        # Filter for pixels inside the cell
        cell_pixel_y = ly[local_cell_mask]
        cell_pixel_x = lx[local_cell_mask]
        pixel_intensities = local_intensity[local_cell_mask]
        
        # Calculate Euclidean distance from center for each pixel
        distances = np.sqrt((cell_pixel_y - local_cy)**2 + (cell_pixel_x - local_cx)**2)
        
        # Normalize distances by the maximum distance in this cell (scale invariance)
        max_dist = np.max(distances)
        if max_dist == 0:
            continue
        norm_distances = distances / max_dist
        
        # Binning: Create radial profile
        # We want the mean intensity in each bin
        # Bins are [0, 1/N, 2/N, ..., 1.0]
        bins = np.linspace(0, 1.0, num_bins + 1)
        
        # Digitizing returns indices 1..num_bins
        bin_indices = np.digitize(norm_distances, bins)
        
        radial_means = []
        for i in range(1, num_bins + 1):
            mask_bin = (bin_indices == i)
            if np.any(mask_bin):
                mean_val = np.mean(pixel_intensities[mask_bin])
                radial_means.append(mean_val)
            else:
                # If a bin is empty (rare with sufficient pixels), we can skip it or interpolate.
                # Skipping is safer for statistics.
                pass
        
        if len(radial_means) < 2:
            continue
            
        radial_means = np.array(radial_means)
        
        # Calculate Coefficient of Variation (CV) of the radial profile
        # CV = std / mean
        prof_mean = np.mean(radial_means)
        prof_std = np.std(radial_means)
        
        if prof_mean > 1e-6:
            cv = prof_std / prof_mean
            cv_values.append(cv)
        else:
            cv_values.append(0.0)

    # 4. Aggregate Results
    if not cv_values:
        return 0.0
        
    result = np.mean(cv_values)
    return float(result)
