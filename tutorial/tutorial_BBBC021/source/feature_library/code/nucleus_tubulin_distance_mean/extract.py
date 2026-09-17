def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    
    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset is (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return 0.0

    # Extract Tubulin channel (Channel 1: Green)
    # Channel mapping: 0=Actin, 1=Tubulin, 2=DAPI
    tubulin_channel = arr[:, :, 1]

    # Handle segmentation masks
    if not segmentation_masks:
        # Without segmentation, we cannot define individual cells to compute this metric
        return 0.0

    # Assign masks based on availability
    # We assume the first mask is Nuclei and the second (if available) is the Whole Cell
    # If only one mask is provided, we use it for both (calculating shift within the nucleus/mask itself)
    nuclei_mask = segmentation_masks[0]
    cell_mask = segmentation_masks[1] if len(segmentation_masks) > 1 else nuclei_mask

    # Ensure masks are labeled integers
    nuclei_mask = nuclei_mask.astype(np.int32)
    cell_mask = cell_mask.astype(np.int32)

    # Get unique labels from the nuclei mask (excluding background 0)
    unique_labels = np.unique(nuclei_mask)
    unique_labels = unique_labels[unique_labels != 0]

    if len(unique_labels) == 0:
        return 0.0

    # Calculate Geometric Centroids of Nuclei
    # We use a dummy array of ones to compute the geometric center of mass
    ones_img = np.ones_like(nuclei_mask, dtype=np.float32)
    nuc_centers = ndimage.center_of_mass(ones_img, labels=nuclei_mask, index=unique_labels)

    # Calculate Intensity-Weighted Centroids of Tubulin
    # We measure this within the 'cell_mask' region for the corresponding label
    # Note: This assumes labels are consistent between masks (Label 1 in Nuclei == Label 1 in Cell)
    tub_centers = ndimage.center_of_mass(tubulin_channel, labels=cell_mask, index=unique_labels)

    # ndimage.center_of_mass returns a tuple if single label, or list of tuples if multiple
    # Normalize to list of tuples for consistent iteration
    if len(unique_labels) == 1:
        nuc_centers = [nuc_centers]
        tub_centers = [tub_centers]

    distances = []

    for nc, tc in zip(nuc_centers, tub_centers):
        # Check for valid centroids (center_of_mass returns NaN if label is missing or sum is 0)
        # This handles cases where a label exists in nuclei_mask but not cell_mask
        if nc is None or tc is None:
            continue
        
        ny, nx = nc
        ty, tx = tc
        
        if np.isnan(ny) or np.isnan(nx) or np.isnan(ty) or np.isnan(tx):
            continue

        # Euclidean distance
        dist = np.sqrt((ny - ty)**2 + (nx - tx)**2)
        distances.append(dist)

    if not distances:
        return 0.0

    # Return the mean distance across all valid cells
    result = np.mean(distances)

    return float(result)
