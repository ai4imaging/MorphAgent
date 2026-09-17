def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from scipy import stats
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed

    # 1. Data Loading and Preprocessing
    # Convert to float32 for calculations
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensions: Expected (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Channel mapping based on dataset description:
    # Channel 0: Actin (Red)
    # Channel 1: Tubulin (Green) - TARGET for intensity
    # Channel 2: DAPI (Blue) - TARGET for nuclei centers
    tubulin_ch = arr[..., 1]
    dapi_ch = arr[..., 2]

    # Normalize Tubulin channel to [0, 1] for consistent slope units
    # Using robust max to avoid hot pixel artifacts
    max_val = np.percentile(tubulin_ch, 99.9)
    if max_val > 0:
        tubulin_norm = tubulin_ch / max_val
    else:
        tubulin_norm = tubulin_ch  # If empty, stays 0

    tubulin_norm = np.clip(tubulin_norm, 0.0, 1.0)

    # 2. Segmentation Logic
    # We need:
    # A. Nuclei centroids (to define r=0)
    # B. Cell masks (to define the boundary of the radial profile)
    
    nuclei_mask = None
    cell_mask = None

    # Check if segmentation masks are provided via arguments
    # The dataset description suggests masks might be available in derived folders,
    # but the function signature allows passing them.
    # If passed, we assume: mask 0 is cells/cytoplasm, mask 1 is nuclei (or check labels/sizes)
    # However, standard practice with *args is to inspect them.
    
    if len(segmentation_masks) > 0:
        # Try to identify nuclei vs cell mask based on typical characteristics if multiple are passed
        # Or assume a specific order. Given the variability, a robust fallback is safer.
        # Let's try to use the first mask as a cell mask.
        # If it's instance segmentation (labeled), great.
        candidate_mask = segmentation_masks[0]
        if candidate_mask.shape == arr.shape[:2]:
            cell_mask = candidate_mask
            # If we have a cell mask, we still need nuclei centroids.
            # We can try to find centroids from the DAPI channel within these cell masks
            # or perform a quick nuclei segmentation.
    
    # Fallback / Refinement Segmentation Pipeline
    # If no masks provided, or to ensure we have matched nuclei/cells:
    if cell_mask is None or nuclei_mask is None:
        # A. Segment Nuclei
        try:
            thresh_nuc = threshold_otsu(dapi_ch)
        except ValueError: # Handle empty images
            thresh_nuc = 0
        
        nuclei_mask_binary = dapi_ch > thresh_nuc
        # Clean up nuclei
        nuclei_mask_binary = ndimage.binary_opening(nuclei_mask_binary, structure=np.ones((3,3)))
        nuclei_labels = label(nuclei_mask_binary)

        # B. Segment Cells (Watershed)
        # Use nuclei as markers
        try:
            thresh_cyto = threshold_otsu(tubulin_ch)
        except ValueError:
            thresh_cyto = 0
            
        # Combine channels for a better cell footprint
        combined_intensity = (tubulin_ch + dapi_ch) / 2.0
        mask_cyto = combined_intensity > (np.mean(combined_intensity) * 0.5) # Loose mask
        
        # Watershed
        # Invert intensity for topographic surface (bright = peaks = basins in inverted)
        # But standard watershed works on gradient or distance. 
        # Here, simple expansion from nuclei is sufficient.
        cell_labels = watershed(-ndimage.gaussian_filter(tubulin_ch, sigma=2), 
                                nuclei_labels, 
                                mask=mask_cyto)
    else:
        # Use provided mask
        cell_labels = cell_mask.astype(int)
        # If we used an external mask, we still need to ensure we can find a "center" for each label.
        # We will calculate the centroid of the mask itself if DAPI isn't aligned, 
        # but ideally we use the intensity-weighted centroid of the DAPI channel within the mask.

    # 3. Feature Extraction: Radial Intensity Slope
    
    props = regionprops(cell_labels, intensity_image=dapi_ch)
    
    slopes = []

    # Grid for distance calculation
    h, w = arr.shape[:2]
    y_indices, x_indices = np.indices((h, w))

    for prop in props:
        # Skip background or tiny artifacts
        if prop.label == 0 or prop.area < 100:
            continue

        # Define the center of the radial profile.
        # Ideally: Centroid of the Nucleus.
        # Fallback: Centroid of the Cell (if nucleus not distinct).
        # Here, we use the weighted centroid of the DAPI intensity within the cell region,
        # which effectively finds the nuclear center even if we only have a whole-cell mask.
        yc, xc = prop.weighted_centroid 
        
        # Get the bounding box to reduce computation
        minr, minc, maxr, maxc = prop.bbox
        
        # Extract local mask and intensity
        local_mask = (cell_labels[minr:maxr, minc:maxc] == prop.label)
        local_tubulin = tubulin_norm[minr:maxr, minc:maxc]
        
        # Calculate distances from centroid for pixels in this cell
        # Adjust indices to be relative to the slice
        local_y = y_indices[minr:maxr, minc:maxc]
        local_x = x_indices[minr:maxr, minc:maxc]
        
        # Euclidean distance
        dists = np.sqrt((local_y - yc)**2 + (local_x - xc)**2)
        
        # Flatten arrays for regression
        # Only consider pixels inside the cell mask
        valid_pixels = local_mask
        if not np.any(valid_pixels):
            continue
            
        pixel_dists = dists[valid_pixels]
        pixel_intensities = local_tubulin[valid_pixels]
        
        # Binning (Sholl Analysis approach) to reduce noise
        # We bin by 1-pixel radius increments
        max_dist = np.max(pixel_dists)
        if max_dist <= 1.0:
            continue
            
        bins = np.arange(0, max_dist + 1, 1.0) # 1-pixel bins
        
        # Compute mean intensity per bin
        # digitize returns indices of bins
        bin_indices = np.digitize(pixel_dists, bins)
        
        # Calculate mean intensity for each radial bin
        # We can use bincount for fast summation
        bin_counts = np.bincount(bin_indices, minlength=len(bins)+1)
        bin_sums = np.bincount(bin_indices, weights=pixel_intensities, minlength=len(bins)+1)
        
        # Avoid division by zero
        valid_bins = bin_counts > 0
        
        radial_means = bin_sums[valid_bins] / bin_counts[valid_bins]
        radial_distances = bins[np.where(valid_bins)[0] - 1] # Adjust index to radius value (approx)
        # Note: digitize index i corresponds to bins[i-1] <= x < bins[i]. 
        # Using the bin edge as distance is sufficient for slope.

        # Need at least 3 points for a meaningful regression
        if len(radial_distances) < 3:
            continue

        # Linear Regression
        # y = slope * x + intercept
        # x = radial_distance
        # y = mean_intensity
        slope, intercept, r_value, p_value, std_err = stats.linregress(radial_distances, radial_means)
        
        # Check for NaN
        if not np.isnan(slope):
            slopes.append(slope)

    # 4. Aggregation
    # Return the median slope across all cells in the image
    # A negative slope is expected (bright center, dark edge).
    # Steeper negative slope (e.g. -0.05) = more perinuclear.
    # Flatter slope (e.g. -0.001) = more diffuse.
    
    if not slopes:
        return 0.0
        
    result = np.median(slopes)
    
    return float(result)
