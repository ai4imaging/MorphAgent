def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from scipy.spatial.distance import cdist  # Corrected import path

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Dataset Dimensions: (512, 512, 3)
    # Channel 2 is DAPI (Nucleus)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Nuclear Channel (Channel 2)
    nuclei_channel = arr[:, :, 2]

    # Normalize
    vmax = np.percentile(nuclei_channel, 99.5) if nuclei_channel.size > 0 else 1.0
    if vmax > 0:
        nuclei_channel = nuclei_channel / vmax
    nuclei_channel = np.clip(nuclei_channel, 0.0, 1.0)

    # Identify Nuclei
    # If segmentation masks are provided, try to find a nuclei mask
    nuclei_labels = None
    
    # Check provided masks first
    if len(segmentation_masks) > 0:
        # Heuristic: Look for the mask with the most objects, assuming nuclei are distinct
        # Or simply use the first one if it looks reasonable
        # Given the dataset description, segmentation masks are likely cell/nuclei labels
        # Let's try to use the first mask if it exists and has labels
        mask = segmentation_masks[0]
        if mask.ndim == 2 and np.max(mask) > 0:
             nuclei_labels = mask.astype(int)

    # Fallback: Compute nuclei segmentation from raw image if no valid mask provided
    if nuclei_labels is None:
        try:
            thresh = threshold_otsu(nuclei_channel)
            binary_nuclei = nuclei_channel > thresh
            # Clean up noise
            binary_nuclei = ndimage.binary_opening(binary_nuclei, structure=np.ones((3,3)))
            nuclei_labels = label(binary_nuclei)
        except Exception:
            return 0.0

    # Extract Centroids
    props = regionprops(nuclei_labels)
    centroids = [p.centroid for p in props]
    
    num_cells = len(centroids)
    
    # If too few cells, clustering coefficient is not well-defined or trivial
    if num_cells < 3:
        return 0.0

    centroids_array = np.array(centroids)

    # Compute Clustering Coefficient based on Nearest Neighbor Distances
    # Clark-Evans Aggregation Index (R) is a standard measure for this.
    # R = Mean_Observed_Distance / Mean_Expected_Distance
    # R < 1 implies clustering, R = 1 implies random, R > 1 implies regularity
    # However, the prompt asks for a "clustering coefficient" where higher means more clustered.
    # We can use the Coefficient of Variation (CV) of the Nearest Neighbor distances,
    # or simply the inverse of the Clark-Evans index, or a ratio of variance to mean.
    
    # Let's implement a metric where higher values indicate stronger clustering.
    # A robust metric is the Coefficient of Variation (CV) of the nearest neighbor distances.
    # In clustered distributions, many points have small NN distances (within cluster), 
    # but some might be isolated, leading to high variance relative to the mean.
    # Alternatively, we can use the ratio of Expected NN distance (random) to Observed NN distance.
    # If cells are clustered, Observed distance is small, so Expected/Observed is large.
    
    # 1. Calculate pairwise distances
    dists = cdist(centroids_array, centroids_array)
    
    # 2. Get Nearest Neighbor distance for each point (exclude self-distance which is 0)
    # Set diagonal to infinity to ignore self
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    
    mean_observed_dist = np.mean(min_dists)
    
    if mean_observed_dist == 0:
        return 10.0 # Maximum clustering (all points on top of each other)

    # 3. Calculate Expected Mean Nearest Neighbor Distance for a random distribution (Poisson)
    # Formula: 0.5 / sqrt(density)
    # Density = N / Area
    area = nuclei_channel.shape[0] * nuclei_channel.shape[1]
    density = num_cells / area
    expected_mean_dist = 0.5 / np.sqrt(density)
    
    # 4. Calculate Index
    # Clark-Evans R = mean_observed_dist / expected_mean_dist
    # If R < 1 (Clustered), we want a high output.
    # If R > 1 (Regular), we want a low output.
    # Let's return the ratio: Expected / Observed.
    # If clustered, Observed is small -> Ratio is High (>1).
    # If random, Ratio ~ 1.
    # If regular/dispersed, Observed is large -> Ratio is Low (<1).
    
    clustering_index = expected_mean_dist / mean_observed_dist

    return float(clustering_index)
