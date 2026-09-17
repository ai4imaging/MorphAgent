def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.morphology import remove_small_objects, binary_opening, disk
    from skimage.measure import label
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    # 1. Check for provided segmentation masks first
    # If a mask is provided, we assume it's a labeled instance mask where each object has a unique ID.
    # We prioritize using the provided mask for consistency if available.
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask = segmentation_masks[0]
        # Handle potential extra dimensions in mask (e.g. if it was loaded as (H, W, 1))
        if mask.ndim == 3:
            mask = mask.squeeze()
        
        # If the mask is already labeled (integer mask with values > 1), count unique labels
        if np.issubdtype(mask.dtype, np.integer) or np.issubdtype(mask.dtype, np.floating):
            # Count unique non-zero labels
            unique_labels = np.unique(mask)
            # Remove 0 (background) if present
            if 0 in unique_labels:
                count = len(unique_labels) - 1
            else:
                count = len(unique_labels)
            return float(count)

    # 2. Fallback: Compute from raw image if no mask is provided
    
    # Convert to appropriate array type
    # The dataset is (512, 512, 3) uint8.
    arr = np.asarray(img)
    
    # Validation: Check shape
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If dimensions don't match expected (H, W, 3), try to adapt or return 0
        if arr.ndim == 2:
            # If 2D, assume it's a single channel image. 
            # However, dataset spec says 3 channels. If we get 2D, we can't be sure it's DAPI.
            # But for robustness, we'll treat it as the intensity image.
            dapi = arr
        else:
            return 0.0
    else:
        # Extract DAPI channel (Channel 2 / Blue)
        # Spec: Channel 0=Actin, 1=Tubulin, 2=DAPI
        dapi = arr[:, :, 2]

    # Convert to float for processing
    dapi = dapi.astype(np.float32)

    # Check for empty image
    if np.max(dapi) == 0:
        return 0.0

    # 3. Preprocessing
    # Gaussian blur to reduce noise and smooth texture within nuclei
    # MCF-7 nuclei can be textured; smoothing helps prevent over-segmentation
    dapi_smooth = ndimage.gaussian_filter(dapi, sigma=2.0)

    # 4. Thresholding
    try:
        thresh_val = threshold_otsu(dapi_smooth)
    except ValueError:
        # Can happen if image is uniform
        return 0.0
        
    binary_mask = dapi_smooth > thresh_val

    # 5. Morphological Cleanup
    # Remove small noise (debris, apoptotic fragments < 50 pixels)
    # MCF-7 cells are relatively large; tiny spots are likely noise or debris
    binary_mask = remove_small_objects(binary_mask, min_size=50)
    
    # Fill holes to ensure solid objects
    binary_mask = ndimage.binary_fill_holes(binary_mask)

    # 6. Separation of Touching Nuclei (Watershed)
    # MCF-7 cells grow in clusters, so nuclei often touch.
    # Simple labeling of the binary mask would undercount.
    
    # Calculate distance transform (distance from background)
    distance = ndimage.distance_transform_edt(binary_mask)
    
    # Find peaks in the distance map (centers of nuclei)
    # min_distance ensures we don't find multiple peaks for one nucleus
    # indices=False returns a boolean mask
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((7, 7)), labels=binary_mask)
    
    # Label the markers
    markers = label(local_maxi)
    
    # Apply watershed
    # -distance is used because watershed works on basins (minima)
    labels = watershed(-distance, markers, mask=binary_mask)

    # 7. Final Count
    # The max label value corresponds to the number of unique regions found
    # (labels are 1, 2, 3, ... N)
    if labels.size == 0:
        return 0.0
        
    count = labels.max()

    return float(count)
