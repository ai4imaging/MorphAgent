def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from skimage.feature import graycomatrix, graycoprops
    from skimage.measure import regionprops, label
    from skimage.filters import threshold_otsu
    from skimage.util import img_as_ubyte
    import warnings

    # Suppress warnings for cleaner output (e.g., potential precision warnings)
    warnings.filterwarnings("ignore")

    # 1. Data Validation and Preparation
    # Ensure input is a numpy array
    img = np.asarray(img)

    # Check dimensionality
    # Expected shape: (512, 512, 3)
    if img.ndim != 3 or img.shape[2] != 3:
        return 0.0

    # Extract Tubulin Channel (Channel 1 - Green)
    # Based on dataset info: Channel 0=Actin, 1=Tubulin, 2=DAPI
    tubulin_channel = img[:, :, 1]

    # 2. Segmentation Handling
    # Determine the mask to use
    labeled_mask = None
    
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the provided segmentation mask
        mask_input = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if mask_input.shape == tubulin_channel.shape:
            # If mask is already labeled (integers > 1), use it directly
            # If mask is binary (0/1 or 0/255), label it
            if np.max(mask_input) > 1:
                labeled_mask = mask_input.astype(int)
            else:
                labeled_mask = label(mask_input > 0)
    
    # Fallback: If no valid mask provided, generate one using Otsu thresholding
    if labeled_mask is None:
        try:
            # Check if image has content
            if np.max(tubulin_channel) == np.min(tubulin_channel):
                return 0.0
            
            thresh = threshold_otsu(tubulin_channel)
            binary_mask = tubulin_channel > thresh
            labeled_mask = label(binary_mask)
        except Exception:
            return 0.0

    # 3. Pre-processing for Texture Analysis
    # Haralick features are sensitive to the number of gray levels.
    # Using full 256 levels on small cell crops results in sparse matrices.
    # We quantize the image to 32 levels (5 bits) for robust texture statistics.
    n_levels = 32
    
    # Normalize to 0-1 range first to handle potential dtype differences
    if tubulin_channel.dtype == np.uint8:
        norm_img = tubulin_channel / 255.0
    else:
        # Robust min-max normalization
        dmin, dmax = np.min(tubulin_channel), np.max(tubulin_channel)
        if dmax > dmin:
            norm_img = (tubulin_channel - dmin) / (dmax - dmin)
        else:
            norm_img = np.zeros_like(tubulin_channel, dtype=float)

    # Quantize to 0-(n_levels-1)
    # We use floor(val * n_levels) and clip to ensure range
    quantized_img = np.floor(norm_img * n_levels).astype(np.uint8)
    quantized_img = np.clip(quantized_img, 0, n_levels - 1)

    # 4. Per-Cell Feature Extraction
    regions = regionprops(labeled_mask, intensity_image=quantized_img)
    
    cell_energies = []

    # GLCM Parameters
    # Distances: 1 pixel (fine texture)
    # Angles: 0, 45, 90, 135 degrees (average for rotational invariance)
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    for region in regions:
        # Skip very small regions that can't support GLCM
        if region.area < 5:
            continue

        # Extract the bounding box image of the cell
        # region.image is the binary mask of the cell within the bounding box
        # region.intensity_image is the quantized intensity within the bounding box
        cell_mask = region.image
        cell_pixels = region.intensity_image

        # We need to compute GLCM only on the pixels belonging to the cell.
        # Standard graycomatrix computes on the rectangular array.
        # Strategy: Set background pixels to a value outside the quantization range (e.g., n_levels),
        # compute GLCM with size (n_levels + 1), then slice the GLCM to exclude the background index.
        
        # Create a working array where background is marked with 'n_levels'
        # cell_pixels values are 0 to 31. Background becomes 32.
        masked_pixels = np.full(cell_pixels.shape, n_levels, dtype=np.uint8)
        masked_pixels[cell_mask] = cell_pixels[cell_mask]

        try:
            # Compute GLCM
            # levels needs to be n_levels + 1 to accommodate the background value
            glcm = graycomatrix(masked_pixels, distances=distances, angles=angles, 
                                levels=n_levels + 1, symmetric=True, normed=False)
            
            # Slice the GLCM to remove the background row/column (index 32)
            # We only want transitions between valid cell pixels (0-31)
            glcm_cell = glcm[0:n_levels, 0:n_levels, :, :]
            
            # Normalize the sliced GLCM
            # Sum over i, j for each distance/angle pair
            glcm_sums = np.sum(glcm_cell, axis=(0, 1), keepdims=True)
            
            # Avoid division by zero
            glcm_sums[glcm_sums == 0] = 1
            glcm_normed = glcm_cell / glcm_sums

            # Calculate Energy (ASM)
            # Energy = sum(p(i,j)^2)
            energy_matrix = graycoprops(glcm_normed, 'energy')
            
            # Average energy across all 4 angles for this cell
            mean_cell_energy = np.mean(energy_matrix)
            cell_energies.append(mean_cell_energy)

        except Exception:
            continue

    # 5. Aggregation
    if not cell_energies:
        return 0.0

    # Return the mean energy across all cells in the image
    result = np.mean(cell_energies)
    
    return float(result)
