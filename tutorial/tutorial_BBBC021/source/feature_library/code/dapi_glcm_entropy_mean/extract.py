def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix
    from skimage.measure import regionprops
    from skimage.filters import threshold_otsu
    from skimage.measure import label as skimage_label
    
    # 1. Extract DAPI Channel (Channel 2 / Blue)
    # Dataset description: (Height, Width, Channels) = (512, 512, 3)
    # Channel 2 is DAPI.
    # Ensure image is at least 3D. If 2D, assume it's single channel or handle gracefully.
    if img.ndim == 3 and img.shape[2] >= 3:
        dapi_channel = img[:, :, 2]
    elif img.ndim == 2:
        # Fallback if only 2D image provided (unlikely given description, but safe)
        dapi_channel = img
    else:
        return 0.0

    # 2. Handle Segmentation
    # If masks are provided, use the first one (assuming it's the primary object/nuclei mask).
    # If not, generate a mask using Otsu thresholding on the DAPI channel.
    labels = None
    if segmentation_masks and len(segmentation_masks) > 0:
        # Use the first available mask
        mask_input = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == dapi_channel.shape[:2]:
            # If the mask is already labeled (int type with values > 1), use it directly
            if np.issubdtype(mask_input.dtype, np.integer) and mask_input.max() > 1:
                labels = mask_input
            else:
                # If binary, label connected components
                labels = skimage_label(mask_input > 0)
    
    # Fallback: Generate segmentation if no valid mask provided
    if labels is None:
        try:
            thresh = threshold_otsu(dapi_channel)
            binary_mask = dapi_channel > thresh
            labels = skimage_label(binary_mask)
        except Exception:
            # Fallback for extremely low contrast/empty images
            return 0.0

    # 3. Preprocessing for GLCM
    # GLCM calculation is expensive and sparse on full 256 levels.
    # Bin the image into fewer levels (e.g., 8 bins) for robust texture statistics.
    n_bins = 8
    # dapi_channel is uint8 (0-255). We bin it to 0-7.
    # We use integer division. 256 / 8 = 32.
    # Pixels 0-31 -> 0, 32-63 -> 1, ..., 224-255 -> 7.
    binned_img = (dapi_channel // (256 // n_bins)).astype(np.uint8)
    
    # Clip just in case to ensure range [0, n_bins-1]
    binned_img = np.clip(binned_img, 0, n_bins - 1)

    # 4. Compute GLCM Entropy per Nucleus
    entropies = []
    
    # Get properties of labeled regions to extract bounding boxes
    props = regionprops(labels, intensity_image=binned_img)
    
    # Define GLCM parameters
    # Distances: 1 pixel
    # Angles: 0, 45, 90, 135 degrees (average for rotational invariance)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    for prop in props:
        # Filter small noise
        if prop.area < 10:
            continue
            
        # Extract the binned intensity patch for this nucleus
        # image_intensity in regionprops returns the intensity image cropped to the bounding box
        # However, it doesn't mask out the background pixels inside the box.
        # We need to use the mask from the region.
        
        patch_intensity = prop.image_intensity  # This is the binned image cropped
        patch_mask = prop.image  # This is the binary mask cropped
        
        # We only want to compute GLCM on the nucleus pixels.
        # GLCM functions usually take a rectangular array.
        # To ignore background, we can't easily pass a mask to graycomatrix in skimage < 0.19 efficiently 
        # without treating background as a specific level.
        # Strategy: 
        # 1. Set background pixels in the patch to a value outside the bin range? No, graycomatrix expects contiguous levels.
        # 2. Standard approach: Compute GLCM on the rectangular patch. 
        #    However, this includes texture boundaries between nucleus and background (0s).
        #    Since we want internal nuclear texture, we should try to minimize background influence.
        #    Given the constraints and standard implementations, computing on the masked rectangular patch 
        #    is the standard approximation. We mask the background to 0 (which is bin 0).
        #    Note: This treats background as "darkest texture".
        
        # Apply mask: keep original binned values where mask is True, else 0
        # Since 0 is a valid bin, this might skew results slightly, but is standard for rectangular GLCM.
        # A more advanced way is to ignore the '0-0' co-occurrence if 0 is background, 
        # but here 0 is also a valid intensity level.
        # For simplicity and robustness in this pipeline, we use the rectangular patch masked.
        
        masked_patch = patch_intensity * patch_mask
        
        # Compute GLCM
        # levels=n_bins (8)
        try:
            glcm = graycomatrix(masked_patch, distances=distances, angles=angles, 
                                levels=n_bins, symmetric=True, normed=True)
        except ValueError:
            continue

        # GLCM shape: (levels, levels, num_distances, num_angles) -> (8, 8, 1, 4)
        
        # Compute Entropy
        # Entropy = - sum(p * log2(p))
        # We compute entropy for each angle, then average.
        
        # Avoid log(0) by adding a tiny epsilon or masking
        p = glcm
        p_normalized = p / (np.sum(p, axis=(0, 1), keepdims=True) + 1e-15)
        
        # Calculate entropy for each angle (last axis)
        # Mask out zeros for log calculation
        mask_p = p_normalized > 0
        entropy_per_angle = -np.sum(
            p_normalized[mask_p] * np.log2(p_normalized[mask_p])
        )
        
        # Since we flattened to compute sum, we need to be careful.
        # Let's do it explicitly per angle to be safe.
        current_cell_entropies = []
        for angle_idx in range(len(angles)):
            matrix = p_normalized[:, :, 0, angle_idx]
            # Filter zeros
            matrix_nonzero = matrix[matrix > 0]
            if matrix_nonzero.size > 0:
                e = -np.sum(matrix_nonzero * np.log2(matrix_nonzero))
                current_cell_entropies.append(e)
            else:
                current_cell_entropies.append(0.0)
        
        # Average entropy across directions for this cell
        mean_cell_entropy = np.mean(current_cell_entropies)
        entropies.append(mean_cell_entropy)

    # 5. Aggregate and Return
    if not entropies:
        return 0.0
        
    result = np.mean(entropies)
    return float(result)
