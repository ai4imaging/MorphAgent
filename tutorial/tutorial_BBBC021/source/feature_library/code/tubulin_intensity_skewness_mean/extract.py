def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import stats
    from scipy import ndimage
    from skimage.measure import regionprops, label
    from skimage.filters import threshold_otsu
    from skimage.morphology import closing, square
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max

    # 1. Data Loading and Preprocessing
    # Convert to float32 for calculations
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not standard (H, W, 3), try to handle potential variations or return 0
        if arr.ndim == 2:
            # If 2D, assume it's a single channel image. 
            # However, dataset spec says 3 channels. If we get 2D, we can't distinguish Tubulin.
            # Return 0.0 as fallback.
            return 0.0
        return 0.0

    # Extract Tubulin Channel (Channel 1 - Green)
    # Channel mapping: 0=Actin, 1=Tubulin, 2=DAPI
    tubulin_channel = arr[:, :, 1]
    
    # Normalize Tubulin channel for segmentation purposes (if needed)
    # But for skewness calculation, we often want the raw distribution shape.
    # However, converting to float is sufficient. We will use `tubulin_channel` for measurements.

    # 2. Segmentation Logic
    # We need a cell mask to measure skewness per cell.
    
    cell_mask = None
    
    # Check if segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Iterate through masks to find a suitable one.
        # Usually, if multiple masks exist, one might be nuclei and one cells.
        # We prefer the one with larger average area or simply the last one if unknown,
        # but often the order is consistent. Let's try to use the first one provided
        # or combine logic if specific metadata was available.
        # Without specific metadata on which is which, we check for a mask that covers a significant portion
        # of the image but not all (background).
        
        # Simple heuristic: Use the first mask provided.
        candidate_mask = segmentation_masks[0]
        
        # Ensure it matches image shape (H, W)
        if candidate_mask.shape == arr.shape[:2]:
            cell_mask = candidate_mask
    
    # Fallback: Generate segmentation if no valid mask provided
    if cell_mask is None:
        # Use DAPI (Channel 2) for nuclei seeding
        dapi_channel = arr[:, :, 2]
        
        # Threshold DAPI
        try:
            thresh_dapi = threshold_otsu(dapi_channel)
            nuclei_mask = dapi_channel > thresh_dapi
        except Exception:
            # If image is empty or uniform
            return 0.0

        # Clean up nuclei
        nuclei_mask = closing(nuclei_mask, square(3))
        
        # Identify markers
        distance = ndimage.distance_transform_edt(nuclei_mask)
        coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=nuclei_mask)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndimage.label(mask)
        
        # Define cell boundaries using Tubulin signal (Channel 1)
        # Tubulin usually fills the cytoplasm
        try:
            thresh_tubulin = threshold_otsu(tubulin_channel)
            # Use a lower threshold to capture faint edges
            foreground_mask = tubulin_channel > (thresh_tubulin * 0.7)
        except Exception:
            foreground_mask = np.ones_like(tubulin_channel, dtype=bool)

        # Watershed
        cell_mask = watershed(-tubulin_channel, markers, mask=foreground_mask)

    # Ensure mask is labeled (integers)
    if cell_mask.dtype == bool:
        cell_mask = label(cell_mask)
    
    # 3. Feature Computation: Skewness per cell
    skewness_values = []
    
    props = regionprops(cell_mask, intensity_image=tubulin_channel)
    
    for prop in props:
        # Filter small artifacts
        if prop.area < 50:
            continue
            
        # Get intensity values for this cell
        # regionprops.image_intensity returns the intensity image cropped to the bounding box
        # We need to mask it with the region's binary image to get only cell pixels
        intensity_crop = prop.image_intensity
        binary_crop = prop.image
        
        if intensity_crop is None or binary_crop is None:
            continue
            
        # Extract valid pixels
        cell_pixels = intensity_crop[binary_crop]
        
        if cell_pixels.size < 10:
            continue
            
        # Check variance to avoid division by zero in skewness calc
        if np.std(cell_pixels) == 0:
            # Uniform region has undefined skewness (or 0 if symmetric, but technically undefined)
            # We skip or treat as 0. Let's treat as 0 (symmetric/flat).
            skewness_values.append(0.0)
            continue
            
        # Calculate skewness
        # bias=False for sample skewness
        skew_val = stats.skew(cell_pixels, bias=False)
        
        # Handle NaN results (e.g. from numerical instability)
        if np.isfinite(skew_val):
            skewness_values.append(skew_val)

    # 4. Aggregation
    if not skewness_values:
        return 0.0
        
    result = np.mean(skewness_values)
    
    return float(result)
