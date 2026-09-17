def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from scipy.stats import entropy

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensions: (512, 512, 3) expected
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Channel mapping based on dataset description
    # Ch 0: Actin (Red)
    # Ch 1: Tubulin (Green) -> Target for radial distribution
    # Ch 2: DAPI (Blue) -> Target for Nucleus center
    tubulin_ch = arr[..., 1]
    dapi_ch = arr[..., 2]

    # Normalize channels
    def normalize(c):
        vmax = np.percentile(c, 99.5) if c.size > 0 else 1.0
        if vmax > 0:
            c = c / vmax
        return np.clip(c, 0.0, 1.0)

    tubulin_norm = normalize(tubulin_ch)
    dapi_norm = normalize(dapi_ch)

    # Identify Nuclei
    # If segmentation masks are provided, use the nuclei mask (likely the one corresponding to DAPI)
    # If not, segment DAPI channel
    nuclei_mask = None
    
    # Heuristic to find nuclei mask in provided masks
    # We look for a mask that overlaps significantly with high DAPI signal or just use the first one if it looks like nuclei
    if len(segmentation_masks) > 0:
        # Check masks. Usually, nuclei are smaller and more numerous or distinct.
        # Without specific metadata on which mask is which, we might fallback to computing it or guessing.
        # However, the prompt implies we should use them if available.
        # Let's try to find a mask that correlates with DAPI intensity.
        best_mask = None
        best_corr = -1
        
        for mask in segmentation_masks:
            if mask.shape == dapi_ch.shape:
                # Simple correlation check: mean DAPI intensity inside mask vs outside
                mask_bool = mask > 0
                if np.sum(mask_bool) > 0:
                    in_mean = np.mean(dapi_ch[mask_bool])
                    out_mean = np.mean(dapi_ch[~mask_bool])
                    contrast = in_mean - out_mean
                    if contrast > best_corr:
                        best_corr = contrast
                        best_mask = mask
        
        if best_mask is not None:
            nuclei_mask = best_mask

    # Fallback: Segment nuclei from DAPI if no good mask found
    if nuclei_mask is None:
        try:
            thresh = threshold_otsu(dapi_norm)
            nuclei_mask = label(dapi_norm > thresh)
        except Exception:
            return 0.0

    # Ensure mask is labeled
    if nuclei_mask.max() == 0:
        return 0.0
    
    if nuclei_mask.dtype == bool:
        nuclei_labels = label(nuclei_mask)
    else:
        nuclei_labels = nuclei_mask

    props = regionprops(nuclei_labels)
    
    # Parameters for radial distribution
    max_radius = 60  # Pixels, reasonable for cell radius in 512x512
    num_bins = 20
    
    entropies = []

    # Iterate over each nucleus to compute radial distribution of tubulin
    for prop in props:
        # Get centroid
        cy, cx = prop.centroid
        
        # Define a bounding box for efficiency
        minr, minc, maxr, maxc = prop.bbox
        
        # Expand bbox to cover max_radius
        minr = max(0, int(cy - max_radius))
        minc = max(0, int(cx - max_radius))
        maxr = min(tubulin_norm.shape[0], int(cy + max_radius + 1))
        maxc = min(tubulin_norm.shape[1], int(cx + max_radius + 1))
        
        # Extract local patch
        local_tubulin = tubulin_norm[minr:maxr, minc:maxc]
        
        # Create coordinate grid relative to centroid
        y_indices, x_indices = np.indices(local_tubulin.shape)
        y_rel = (y_indices + minr) - cy
        x_rel = (x_indices + minc) - cx
        
        # Compute distances
        distances = np.sqrt(y_rel**2 + x_rel**2)
        
        # Mask for valid radius
        valid_mask = distances <= max_radius
        
        if np.sum(valid_mask) == 0:
            continue
            
        valid_distances = distances[valid_mask]
        valid_intensities = local_tubulin[valid_mask]
        
        # Bin the distances
        # We want the distribution of intensity *over distance*.
        # i.e., how much intensity is at radius r?
        
        # Create bins
        bins = np.linspace(0, max_radius, num_bins + 1)
        
        # Digitize distances to find which bin they belong to
        bin_indices = np.digitize(valid_distances, bins) - 1
        
        # Sum intensities per bin
        bin_intensities = np.zeros(num_bins)
        for i in range(num_bins):
            # Sum intensity for pixels in this radial shell
            mask_bin = (bin_indices == i)
            if np.any(mask_bin):
                bin_intensities[i] = np.sum(valid_intensities[mask_bin])
                
                # Normalize by area of the shell (number of pixels) to get mean intensity?
                # The feature description says "radial distribution of tubulin intensity".
                # Usually, this implies the profile of density or total amount.
                # If we want "disorder", a flat distribution (uniform intensity everywhere) has high entropy.
                # A structured distribution (e.g., ring or peak) has lower entropy.
                # However, the number of pixels increases with radius (2*pi*r).
                # If tubulin is uniformly distributed in space, total intensity increases linearly with r.
                # To measure the "structure" of the cytoskeleton, we usually look at mean intensity per radial bin.
                # Let's normalize by pixel count in the bin to get average density at that radius.
                pixel_count = np.sum(mask_bin)
                if pixel_count > 0:
                    bin_intensities[i] /= pixel_count
        
        # Now we have a radial profile of mean intensities.
        # Calculate entropy of this profile.
        # To calculate entropy, we treat the profile as a probability mass function (PMF).
        # We must normalize so sum(p) = 1.
        
        total_intensity = np.sum(bin_intensities)
        if total_intensity > 0:
            pmf = bin_intensities / total_intensity
            # Calculate Shannon entropy
            # entropy() function from scipy.stats uses base e by default.
            # We can use base 2 for bits, but relative values matter most.
            # Using base e is standard.
            ent = entropy(pmf)
            entropies.append(ent)

    if not entropies:
        return 0.0

    # Return the mean entropy across all cells in the image
    result = np.mean(entropies)
    
    return float(result)
