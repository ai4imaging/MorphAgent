def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.measure import regionprops
    from skimage.filters import threshold_otsu
    from skimage.morphology import label

    # 1. Input Validation and Channel Selection
    # Dataset is (512, 512, 3), DAPI is Channel 2 (Blue)
    img_arr = np.asarray(img)
    
    if img_arr.ndim != 3 or img_arr.shape[2] != 3:
        return 0.0
    
    # Extract DAPI channel
    dapi_channel = img_arr[:, :, 2]

    # 2. Determine Mask
    # If segmentation masks are provided, use the first one (assuming it's nuclei/cells).
    # If not, generate a mask using Otsu thresholding on the DAPI channel.
    mask = None
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Check if the mask is valid
        input_mask = segmentation_masks[0]
        if input_mask.shape == dapi_channel.shape:
            mask = input_mask > 0 # Convert to boolean
    
    if mask is None:
        # Fallback: Create mask from DAPI
        try:
            thresh = threshold_otsu(dapi_channel)
            mask = dapi_channel > thresh
        except Exception:
            # Fallback for empty/uniform images
            return 0.0

    # If mask is empty, return 0
    if not np.any(mask):
        return 0.0

    # 3. Preprocessing for GLCM
    # GLCM requires integer types. We need to bin the image to reduce levels (e.g., 64 or 256).
    # Using 64 levels is often robust and faster.
    n_levels = 64
    
    # Mask the DAPI channel: Set background to 0, but we need to be careful.
    # Standard approach: Compute GLCM only on pixels within the mask.
    # However, skimage.feature.graycomatrix computes on the whole rectangular array.
    # To handle irregular shapes (nuclei), we can:
    # A. Compute GLCM on the whole image but ignore the background index (0) if we set background to 0.
    # B. Extract bounding boxes of individual nuclei and compute per nucleus.
    
    # Approach B is more biologically accurate for "mean entropy within nuclei".
    # We will label the mask and iterate through regions.
    
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask, intensity_image=dapi_channel)
    
    if not regions:
        return 0.0

    entropies = []

    # Parameters for GLCM
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    for region in regions:
        # Extract the intensity patch for the single nucleus
        # region.image is the binary mask of the object in the bounding box
        # region.intensity_image is the intensity values in the bounding box
        
        # We need to quantize the intensity image to [0, n_levels-1]
        # Normalization should be consistent. We can normalize per cell or globally.
        # Global normalization is usually better for comparing textures across images.
        # However, here we are working with uint8 input.
        
        # Let's use the raw uint8 values but bin them to 64 levels for the GLCM.
        # Input is 0-255. 
        # bin_index = value // (256 // n_levels)
        
        intensity_patch = region.intensity_image
        mask_patch = region.image
        
        # Skip very small regions
        if intensity_patch.size < 10:
            continue

        # Quantize
        # 256 / 64 = 4. So we divide by 4.
        quantized_patch = (intensity_patch // (256 // n_levels)).astype(np.uint8)
        
        # We need to handle the background within the bounding box.
        # graycomatrix doesn't support a mask directly.
        # We can set background pixels to a specific value (e.g., n_levels) 
        # and then ignore that row/col in the GLCM, OR just compute on the rect
        # and accept slight boundary artifacts (common in high-throughput).
        # Better approach for accuracy: Set background to -1 (or a unique value) 
        # but GLCM requires positive integers.
        
        # Let's set background to 0 and shift actual values to 1..n_levels.
        # Then we ignore row/col 0 in the GLCM.
        
        # Shift values: 0..63 -> 1..64
        quantized_patch_shifted = quantized_patch + 1
        
        # Apply mask: background becomes 0
        quantized_patch_masked = quantized_patch_shifted * mask_patch
        
        # Compute GLCM
        # levels needs to be n_levels + 1 to accommodate the 0 (background) and 1..64 (signal)
        try:
            glcm = graycomatrix(quantized_patch_masked, distances=distances, angles=angles, 
                                levels=n_levels + 1, symmetric=True, normed=True)
        except ValueError:
            continue
            
        # glcm shape: (levels, levels, num_distances, num_angles)
        # We want to remove the background interactions (index 0).
        # The background (0) interacts with itself (0,0) and with object boundaries (0, x).
        # We slice the GLCM to exclude the 0-th row and 0-th column.
        glcm_cut = glcm[1:, 1:, :, :]
        
        # Re-normalize the GLCM so it sums to 1
        # Sum over the first two dimensions (i, j)
        glcm_sums = np.sum(glcm_cut, axis=(0, 1), keepdims=True)
        
        # Avoid division by zero
        glcm_sums[glcm_sums == 0] = 1.0
        glcm_norm = glcm_cut / glcm_sums
        
        # Compute Entropy
        # Entropy = - sum(p * log2(p + epsilon))
        # We compute entropy for each (distance, angle) pair
        
        p = glcm_norm
        # Mask out zeros for log calculation
        mask_p = p > 0
        p_masked = p[mask_p]
        
        # If the patch was effectively empty after masking
        if p_masked.size == 0:
            continue
            
        # Calculate entropy for the whole block? No, we need it per (dist, angle).
        # Let's iterate or use vectorized ops carefully.
        
        # Vectorized entropy calculation per (dist, angle)
        # p is (64, 64, 1, 4)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            log_p = np.log2(p)
            log_p[~mask_p] = 0 # log(0) -> -inf, handled by mask
            term = p * log_p
        
        # Sum over i, j
        entropy_per_angle = -np.sum(term, axis=(0, 1))
        
        # Average over angles and distances for this cell
        mean_cell_entropy = np.mean(entropy_per_angle)
        entropies.append(mean_cell_entropy)

    if not entropies:
        return 0.0

    # 4. Aggregate
    # Return the mean entropy across all nuclei
    result = np.mean(entropies)

    return float(result)
