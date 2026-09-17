def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import regionprops, label
    from skimage.filters import threshold_otsu
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    # 1. Data Loading and Preprocessing
    # Convert to float32 for precision
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality and validity
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Channels
    # Channel 1: Tubulin (Green) - Signal to measure
    # Channel 2: DAPI (Blue) - Reference for center (Nucleus)
    tubulin = arr[..., 1]
    dapi = arr[..., 2]

    # Normalize channels to [0, 1] for consistent thresholding/weighting
    def normalize_ch(ch):
        v_min, v_max = ch.min(), ch.max()
        if v_max - v_min > 1e-6:
            return (ch - v_min) / (v_max - v_min)
        return ch

    tubulin_norm = normalize_ch(tubulin)
    dapi_norm = normalize_ch(dapi)

    # 2. Segmentation Logic
    # We need:
    #   A. Nuclear centroids (to define r=0)
    #   B. Cell masks (to define the domain of the tubulin signal for that nucleus)

    nuclei_mask = None
    cell_mask = None

    # Handle provided masks
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        nuclei_mask = segmentation_masks[0]
        # If a second mask is provided, assume it's the cell/cytoplasm mask
        if len(segmentation_masks) > 1 and segmentation_masks[1] is not None:
            cell_mask = segmentation_masks[1]
    
    # Fallback: Generate Nuclei Mask if missing
    if nuclei_mask is None:
        try:
            thresh_val = threshold_otsu(dapi_norm)
            nuclei_bool = dapi_norm > thresh_val
            # Clean up noise
            nuclei_bool = ndimage.binary_opening(nuclei_bool, structure=np.ones((3,3)))
            nuclei_mask = label(nuclei_bool)
        except Exception:
            return 0.0

    # Fallback: Generate Cell Mask if missing (using Watershed)
    if cell_mask is None:
        try:
            # Create a background/foreground marker
            # Combine channels to find general cellular area
            combined_signal = (tubulin_norm + dapi_norm) / 2.0
            try:
                bg_thresh = threshold_otsu(combined_signal)
            except:
                bg_thresh = 0.1
            
            foreground_mask = combined_signal > bg_thresh
            
            # Seed watershed with nuclei
            # We need to ensure nuclei_mask is integer labels
            markers = nuclei_mask.astype(np.int32)
            
            # Compute distance transform for topology
            # Invert intensity for watershed (it floods basins)
            # We use the inverse of the signal as the "elevation map"
            elevation_map = -combined_signal
            
            cell_mask = watershed(elevation_map, markers, mask=foreground_mask)
        except Exception:
            # If watershed fails, just use nuclei mask as cell mask (worst case)
            cell_mask = nuclei_mask

    # 3. Feature Extraction: Radial Distribution
    
    # Get properties of nuclei to find centroids
    nuclei_props = regionprops(nuclei_mask)
    # Map label ID to centroid (row, col)
    nuclei_centroids = {p.label: p.centroid for p in nuclei_props}

    # Get properties of cells to iterate over pixels
    cell_props = regionprops(cell_mask, intensity_image=tubulin_norm)

    peak_values = []

    for cell in cell_props:
        label_id = cell.label
        
        # We need the corresponding nucleus centroid
        if label_id not in nuclei_centroids:
            continue
            
        nuc_r, nuc_c = nuclei_centroids[label_id]
        
        # Get coordinates of all pixels in this cell
        # coords returns (row, col) array of shape (N, 2)
        pixel_coords = cell.coords
        if pixel_coords.shape[0] < 10: # Skip tiny fragments
            continue

        # Get intensities of these pixels
        # image_intensity in regionprops corresponds to the intensity_image passed (tubulin_norm)
        # However, regionprops crops the image. It's safer to index the original array using coords.
        pixel_intensities = tubulin_norm[pixel_coords[:, 0], pixel_coords[:, 1]]

        # Calculate Euclidean distance from nucleus centroid to each pixel
        # d = sqrt((r - r0)^2 + (c - c0)^2)
        dists = np.sqrt((pixel_coords[:, 0] - nuc_r)**2 + (pixel_coords[:, 1] - nuc_c)**2)

        # Compute Radial Profile
        # We bin the distances. 
        # Bin size: 2 pixels is a reasonable trade-off between resolution and noise.
        bin_size = 2.0
        max_dist = np.max(dists)
        if max_dist == 0:
            continue
            
        bins = np.arange(0, max_dist + bin_size, bin_size)
        
        # Weighted histogram (sum of intensities per bin)
        intensity_sum, _ = np.histogram(dists, bins=bins, weights=pixel_intensities)
        
        # Count histogram (number of pixels per bin)
        pixel_count, _ = np.histogram(dists, bins=bins)
        
        # Avoid division by zero
        valid_bins = pixel_count > 0
        
        if not np.any(valid_bins):
            continue

        # Mean Radial Intensity Profile
        # We calculate mean intensity per ring to normalize for the increasing area of outer rings.
        # If we just summed, outer rings would naturally have more signal just because they are bigger.
        radial_profile = np.zeros_like(intensity_sum)
        radial_profile[valid_bins] = intensity_sum[valid_bins] / pixel_count[valid_bins]
        
        # Normalize the profile itself to be a distribution (sum to 1) or scale-invariant
        # Here, we want to detect "peakedness". 
        # If we normalize by sum, a sharp peak will have a high value.
        # If we normalize by max, the peak is always 1.0, which defeats the purpose.
        # Normalizing by sum makes it a probability distribution of "where the mean intensity is concentrated".
        profile_sum = np.sum(radial_profile)
        
        if profile_sum > 0:
            normalized_profile = radial_profile / profile_sum
            
            # The feature is the PEAK of this distribution.
            # A collapsed cytoskeleton (perinuclear) will have a high peak at low radii.
            # A spread cytoskeleton will have a flatter distribution (lower peak).
            peak_val = np.max(normalized_profile)
            peak_values.append(peak_val)

    # 4. Aggregation
    if not peak_values:
        return 0.0

    result = np.mean(peak_values)
    
    return float(result)
