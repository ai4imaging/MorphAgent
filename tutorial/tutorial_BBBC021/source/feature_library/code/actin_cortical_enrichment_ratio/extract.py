def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.morphology import binary_erosion, disk
    from skimage.filters import threshold_otsu

    # 1. Data Loading and Preprocessing
    # Convert to float32 for precision
    arr = np.asarray(img, dtype=np.float32)
    
    # Check dimensionality
    if arr.ndim != 3 or arr.shape[2] != 3:
        # If not standard 3-channel image, try to adapt or return 0
        if arr.ndim == 2:
            # If 2D, assume it's a single channel image, treat as Actin if it's the only data
            actin_channel = arr
        else:
            return 0.0
    else:
        # Standard (H, W, C) = (512, 512, 3)
        # Channel 0 is Actin (Red)
        actin_channel = arr[:, :, 0]

    # Normalize intensity to prevent overflow issues, though ratios are relative
    # We use raw values for ratios usually, but ensuring float is enough.
    # Clip to avoid negative values if any preprocessing happened outside
    actin_channel = np.clip(actin_channel, 0, None)

    # 2. Segmentation Handling
    # We need a cell mask to define the boundary vs center
    labels = None
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use provided segmentation mask
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape == actin_channel.shape:
            labels = mask_input.astype(int)
        elif mask_input.ndim == 3 and mask_input.shape[:2] == actin_channel.shape:
             # If mask is 3D (e.g. one-hot or labeled stack), take max projection or first channel
             labels = np.max(mask_input, axis=2).astype(int)
    
    # Fallback: Generate segmentation if none provided or invalid
    if labels is None:
        # Simple segmentation on Actin channel
        # Smooth to reduce noise
        smooth = ndimage.gaussian_filter(actin_channel, sigma=2)
        try:
            thresh = threshold_otsu(smooth)
            binary_mask = smooth > thresh
            # Fill holes
            binary_mask = ndimage.binary_fill_holes(binary_mask)
            # Label connected components
            labels = label(binary_mask)
        except Exception:
            # If thresholding fails (e.g. empty image), return 0
            return 0.0

    # 3. Feature Computation: Cortical Enrichment Ratio
    # Ratio = Mean Intensity (Cortex) / Mean Intensity (Center)
    
    # Parameters
    # Radius for erosion to define "cortex" width. 
    # For 512x512 MCF-7 cells, a 3-5 pixel rim is reasonable.
    erosion_radius = 4
    selem = disk(erosion_radius)
    
    ratios = []
    
    # Iterate over each cell
    props = regionprops(labels, intensity_image=actin_channel)
    
    for prop in props:
        # Get the bounding box image of the cell mask
        # This is faster than processing the whole 512x512 array for each cell
        cell_mask_local = prop.image  # Binary mask of the cell in bbox
        cell_intensity_local = prop.intensity_image # Intensity of the cell in bbox
        
        # Skip very small cells where erosion would consume the whole cell
        if np.sum(cell_mask_local) < (erosion_radius * 2)**2:
            continue

        # Define Center (Inner Cytoplasm) via erosion
        # We pad the local mask slightly to handle border effects correctly during erosion if needed,
        # but regionprops usually handles bbox well.
        center_mask_local = binary_erosion(cell_mask_local, selem)
        
        # Define Cortex (Boundary) as Cell Mask - Center Mask
        cortex_mask_local = np.logical_xor(cell_mask_local, center_mask_local)
        
        # Check if we have valid pixels for both regions
        if np.sum(center_mask_local) == 0 or np.sum(cortex_mask_local) == 0:
            continue
            
        # Compute Mean Intensities
        mean_cortex = np.mean(cell_intensity_local[cortex_mask_local])
        mean_center = np.mean(cell_intensity_local[center_mask_local])
        
        # Compute Ratio
        # Add epsilon to denominator to avoid division by zero
        epsilon = 1e-7
        ratio = mean_cortex / (mean_center + epsilon)
        
        ratios.append(ratio)

    # 4. Aggregation
    if not ratios:
        return 0.0
        
    # Return the mean ratio across the population
    result = np.mean(ratios)
    
    return float(result)
