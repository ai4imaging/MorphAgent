def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk
    from skimage.segmentation import watershed

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Expected: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Channels
    # Channel 0: Actin (Red) - Cytoskeleton/Cell Body
    # Channel 2: DAPI (Blue) - Nucleus
    actin_channel = arr[:, :, 0]
    dapi_channel = arr[:, :, 2]

    # Normalize channels to [0, 1]
    def normalize(c):
        vmax = np.percentile(c, 99.5) if c.size > 0 else 1.0
        if vmax <= 0: vmax = 1.0
        return np.clip(c / vmax, 0.0, 1.0)

    actin_norm = normalize(actin_channel)
    dapi_norm = normalize(dapi_channel)

    # --- Segmentation Logic ---
    # We need matched instance segmentation: Label X in nuclei mask must correspond to Label X in cell mask.
    # Since provided masks might not guarantee this 1-to-1 mapping (or might be missing),
    # we perform a robust Nucleus-Seeded Watershed segmentation to ensure correspondence.

    # 1. Segment Nuclei (Seeds)
    try:
        thresh_dapi = threshold_otsu(dapi_norm)
    except ValueError:  # Handle empty images
        return 0.0
    
    nuclei_mask = dapi_norm > thresh_dapi
    # Clean up noise
    nuclei_mask = binary_opening(nuclei_mask, footprint=disk(2))
    nuclei_labels = label(nuclei_mask)

    if nuclei_labels.max() == 0:
        return 0.0

    # 2. Segment Cell Body (Basin/Mask)
    try:
        thresh_actin = threshold_otsu(actin_norm)
    except ValueError:
        thresh_actin = 0.1
    
    # The cell mask defines the boundaries where the watershed can grow
    # We combine actin and dapi signal for the foreground to ensure nuclei are included
    foreground_mask = (actin_norm > thresh_actin) | nuclei_mask
    
    # 3. Watershed to define Cell Boundaries
    # We use the inverse of the actin intensity as the "elevation map" so water flows to edges
    # We seed with nuclei labels
    cell_labels = watershed(-actin_norm, nuclei_labels, mask=foreground_mask)

    # --- Feature Computation ---
    # Calculate centroids for nuclei and cells
    nuclei_props = regionprops(nuclei_labels)
    cell_props = regionprops(cell_labels)

    # Create a lookup for cell centroids by label
    # Note: regionprops are not guaranteed to be sorted by label, so we map them
    cell_centroids = {prop.label: prop.centroid for prop in cell_props}

    distances = []

    for n_prop in nuclei_props:
        label_id = n_prop.label
        
        # Check if this nucleus has a corresponding cell body
        if label_id in cell_centroids:
            n_cent = np.array(n_prop.centroid)
            c_cent = np.array(cell_centroids[label_id])
            
            # Euclidean distance between Nucleus Centroid and Cell Centroid
            dist = np.linalg.norm(n_cent - c_cent)
            distances.append(dist)

    # --- Aggregation ---
    if not distances:
        return 0.0

    result = np.mean(distances)

    return float(result)
