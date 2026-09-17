def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.segmentation import find_boundaries
    from skimage.filters import sobel
    
    # 1. Input Validation and Setup
    # Check if segmentation masks are provided
    if not segmentation_masks or len(segmentation_masks) == 0:
        return 0.0
    
    # The first mask is typically the nuclear mask in this dataset context
    # (or at least the most relevant one for a nuclear feature)
    nuclear_mask = segmentation_masks[0]
    
    # Check image dimensions
    # Expected shape is (512, 512, 3)
    if img.ndim != 3 or img.shape[2] != 3:
        return 0.0
        
    # 2. Channel Selection and Preprocessing
    # Channel 2 is DAPI (Blue) -> Nucleus
    dapi_channel = img[:, :, 2]
    
    # Normalize to [0, 1] float
    # The input is uint8 (0-255)
    dapi_norm = dapi_channel.astype(np.float32) / 255.0
    
    # 3. Compute Global Gradient Magnitude
    # We compute the gradient magnitude of the entire DAPI channel first.
    # The Sobel filter is a standard way to approximate the gradient magnitude.
    # High values indicate sharp transitions (edges).
    gradient_mag = sobel(dapi_norm)
    
    # 4. Process Individual Nuclei
    # We need to iterate through labeled regions to compute the metric per cell.
    # This avoids background noise or non-cellular gradients affecting the score.
    
    # Ensure mask is integer type for label processing
    nuclear_mask = nuclear_mask.astype(int)
    unique_labels = np.unique(nuclear_mask)
    
    # Remove background label (0)
    unique_labels = unique_labels[unique_labels > 0]
    
    if len(unique_labels) == 0:
        return 0.0
    
    per_cell_gradients = []
    
    for label_id in unique_labels:
        # Create a binary mask for the current nucleus
        # Optimization: We can slice the bounding box to speed up processing for large images,
        # but for 512x512, full mask operations are acceptable and safer.
        cell_mask = (nuclear_mask == label_id)
        
        # 5. Define the Boundary
        # We want the gradient AT the boundary.
        # mode='thick' returns pixels both inside and outside the contour, 
        # capturing the transition zone where the gradient is most relevant.
        # mode='inner' would only capture the edge inside the nucleus.
        # mode='outer' would only capture the edge outside.
        # 'thick' is generally best for capturing the full slope of the edge.
        boundary_mask = find_boundaries(cell_mask, mode='thick')
        
        # 6. Extract Gradient Values
        # Get the gradient magnitude values specifically at the boundary pixels
        boundary_gradients = gradient_mag[boundary_mask]
        
        if boundary_gradients.size > 0:
            # Compute mean gradient for this specific nucleus
            mean_grad = np.mean(boundary_gradients)
            per_cell_gradients.append(mean_grad)
            
    # 7. Aggregation
    # If we found valid cells, return the average of their boundary gradients.
    if len(per_cell_gradients) > 0:
        result = np.mean(per_cell_gradients)
    else:
        result = 0.0
        
    return float(result)
