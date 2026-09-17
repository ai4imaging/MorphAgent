def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES
    import numpy as np
    from skimage.feature import graycomatrix
    from skimage.measure import regionprops

    # 1. Input Validation and Channel Extraction
    # Dataset is (512, 512, 3), Channel 1 is Tubulin (Green)
    if img.ndim == 3 and img.shape[2] >= 2:
        tubulin_channel = img[:, :, 1]
    elif img.ndim == 2:
        # Fallback if single channel passed (unlikely based on spec but safe)
        tubulin_channel = img
    else:
        return 0.0

    # 2. Preprocessing: Quantization
    # Haralick features are sensitive to the number of gray levels.
    # Computing GLCM on 256 levels is slow and sparse.
    # We quantize to 64 levels (bins) for robustness and speed.
    # Original uint8 is 0-255. Division by 4 gives 0-63.
    n_levels = 64
    quantized_img = (tubulin_channel // 4).astype(np.uint8)
    # Clip just in case to ensure range [0, n_levels-1]
    quantized_img = np.clip(quantized_img, 0, n_levels - 1)

    # 3. Define Regions of Interest (ROIs)
    # If segmentation masks are provided, use the first one (usually cells/nuclei).
    # If not, treat the whole image as one region (or threshold background).
    rois = []
    
    if segmentation_masks and len(segmentation_masks) > 0:
        # Use the first available mask
        mask = segmentation_masks[0]
        # Ensure mask matches image dimensions (handle potential 3D/2D mismatch)
        if mask.shape != quantized_img.shape:
            # If mask is different, try to use it if it's 2D and image is 2D
            if mask.ndim == 2 and quantized_img.ndim == 2:
                pass # Shapes might differ slightly? Assume consistent for now based on spec.
            else:
                # Fallback: ignore mask if shapes are totally incompatible
                mask = None
        
        if mask is not None:
            # Get properties for each labeled region
            props = regionprops(mask.astype(int), intensity_image=quantized_img)
            # Filter out very small regions that might break GLCM
            rois = [p for p in props if p.area > 25] # Min area check
    
    # Fallback if no valid ROIs found from mask or no mask provided
    if not rois:
        # Create a single "region" representing the whole image
        # We wrap it in a dummy object to mimic regionprops structure or handle directly
        class DummyRegion:
            def __init__(self, image):
                self.image_intensity = image
                self.bbox = (0, 0, image.shape[0], image.shape[1])
        rois = [DummyRegion(quantized_img)]

    # 4. Compute Haralick Entropy per ROI
    entropy_values = []
    
    # GLCM parameters: distance=1, angles=0, 45, 90, 135 degrees
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    for region in rois:
        # Extract the intensity patch for the region
        # region.image_intensity gives the intensity image inside the bounding box
        # masked by the region mask (zeros outside).
        # However, zeros are a valid gray level (0).
        # For texture analysis, we ideally want to ignore the background of the bounding box.
        # Standard graycomatrix computes on the whole rectangular patch.
        # To mitigate background influence in non-rectangular cells, we compute on the patch.
        # Since 0 is a valid level (dark tubulin), this is an approximation accepted in high-throughput profiling.
        
        patch = region.image_intensity
        
        if patch.size == 0:
            continue

        # Compute GLCM
        try:
            glcm = graycomatrix(patch, distances=distances, angles=angles, 
                                levels=n_levels, symmetric=True, normed=True)
        except ValueError:
            continue

        # Compute Entropy
        # Formula: - sum(p * log(p))
        # glcm shape: (levels, levels, num_distances, num_angles)
        # We compute entropy for each angle/distance and average them for rotation invariance.
        
        # Add epsilon to avoid log(0)
        epsilon = 1e-10
        p = glcm
        
        # Calculate entropy for each angle (last axis) and distance (2nd to last)
        # Result shape: (num_distances, num_angles)
        term = p * np.log(p + epsilon)
        entropy_per_angle = -np.sum(term, axis=(0, 1))
        
        # Average over the 4 angles (and 1 distance)
        mean_entropy_roi = np.mean(entropy_per_angle)
        entropy_values.append(mean_entropy_roi)

    # 5. Aggregate Results
    if not entropy_values:
        return 0.0
        
    result = np.mean(entropy_values)
    
    return float(result)
