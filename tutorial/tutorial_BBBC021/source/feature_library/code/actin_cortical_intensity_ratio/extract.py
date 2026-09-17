def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_erosion, disk, binary_closing, remove_small_objects

    # 1. Data Loading and Preprocessing
    # Ensure image is float for calculations
    arr = np.asarray(img, dtype=np.float32)

    # Check dimensionality and extract Actin channel (Channel 0)
    # Expected shape: (512, 512, 3)
    if arr.ndim == 3 and arr.shape[2] >= 1:
        actin_channel = arr[..., 0]  # Channel 0 is Actin (Red)
    elif arr.ndim == 2:
        # Fallback if single channel passed (unlikely based on spec but safe)
        actin_channel = arr
    else:
        return 0.0

    # Normalize actin channel to [0, 1] range for stability
    # Use robust max to avoid hot pixel scaling issues
    vmax = np.percentile(actin_channel, 99.5)
    if vmax > 0:
        actin_channel = actin_channel / vmax
    actin_channel = np.clip(actin_channel, 0.0, 1.0)

    # 2. Mask Generation / Handling
    mask = None
    
    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask (assumed to be cell/cytoplasm mask)
        input_mask = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if input_mask.shape[:2] == actin_channel.shape[:2]:
            mask = input_mask > 0
    
    # Fallback: Generate mask from Actin channel if no external mask provided
    if mask is None:
        # Gaussian blur to smooth noise before thresholding
        blurred = ndimage.gaussian_filter(actin_channel, sigma=2)
        try:
            thresh = threshold_otsu(blurred)
            mask = blurred > thresh
            # Morphological cleanup
            mask = binary_closing(mask, disk(3))
            mask = remove_small_objects(mask, min_size=50)
        except Exception:
            # Fallback for extremely low signal images where otsu fails
            return 0.0

    # 3. Instance Labeling
    # We compute the ratio per cell to avoid bias from large/bright cells
    labeled_cells = label(mask)
    regions = regionprops(labeled_cells, intensity_image=actin_channel)

    if not regions:
        return 0.0

    # 4. Feature Computation: Cortical vs Inner Intensity Ratio
    ratios = []
    
    # Define erosion radius for cortex definition (3 pixels ~ thin boundary at 512x512)
    erosion_radius = 3
    selem = disk(erosion_radius)
    epsilon = 1e-6  # To prevent division by zero

    for region in regions:
        # Extract the bounding box image for the current cell to speed up morphology
        # region.image is the binary mask of the cell in the bbox
        cell_mask_local = region.image
        
        # Skip very small cells that would disappear after erosion
        if np.sum(cell_mask_local) < (erosion_radius * 2)**2:
            continue

        # Define Inner Cytoplasm: Eroded version of the cell mask
        # We pad the local mask slightly to handle boundary erosion correctly within the bbox context
        padded_mask = np.pad(cell_mask_local, erosion_radius + 1, mode='constant', constant_values=0)
        eroded_padded = binary_erosion(padded_mask, selem)
        
        # Crop back to original bbox size
        inner_mask_local = eroded_padded[erosion_radius+1 : -erosion_radius-1, 
                                         erosion_radius+1 : -erosion_radius-1]

        # Define Cortex: The pixels removed by erosion (Cell - Inner)
        cortex_mask_local = np.logical_and(cell_mask_local, ~inner_mask_local)

        # Get intensity values from the original image using the bbox
        # region.intensity_image gives the intensity crop
        intensity_local = region.intensity_image

        # Check if we have valid pixels in both regions
        if np.sum(inner_mask_local) == 0 or np.sum(cortex_mask_local) == 0:
            continue

        # Calculate Mean Intensities
        mean_cortex = np.mean(intensity_local[cortex_mask_local])
        mean_inner = np.mean(intensity_local[inner_mask_local])

        # Compute Ratio
        # High ratio (>1) -> Cortical ring / Rounding
        # Low ratio (~1 or <1) -> Diffuse / Stress fibers
        ratio = mean_cortex / (mean_inner + epsilon)
        ratios.append(ratio)

    # 5. Aggregation
    if not ratios:
        return 0.0

    # Return the median ratio across the cell population
    # Median is robust to outliers (e.g., segmentation errors or debris)
    result = np.median(ratios)

    return float(result)
