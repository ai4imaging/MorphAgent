def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import binary_erosion, disk
    from skimage.measure import label
    from skimage.filters import threshold_otsu

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3), uint8
    # Channel 2 is DAPI (Blue)
    if arr.ndim == 3 and arr.shape[2] == 3:
        dapi_channel = arr[:, :, 2]
    elif arr.ndim == 2:
        # Fallback if only one channel is passed, though dataset spec says 3
        dapi_channel = arr
    else:
        return 0.0

    # Check for valid image data
    if dapi_channel.size == 0 or np.max(dapi_channel) == 0:
        return 0.0

    # Determine Segmentation Mask
    # Priority: Use provided segmentation mask, otherwise generate one
    mask = None
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        # Use the first provided mask
        input_mask = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if input_mask.shape[:2] == dapi_channel.shape[:2]:
            mask = input_mask > 0
    
    # Fallback: Generate mask if none provided or invalid
    if mask is None:
        try:
            # Simple preprocessing to reduce noise
            blurred = ndimage.gaussian_filter(dapi_channel, sigma=2.0)
            thresh = threshold_otsu(blurred)
            mask = blurred > thresh
            # Fill holes to ensure solid nuclei
            mask = ndimage.binary_fill_holes(mask)
        except Exception:
            return 0.0

    # Ensure mask is boolean
    mask = mask.astype(bool)
    
    # If mask is empty, return 0.0
    if not np.any(mask):
        return 0.0

    # Label the objects to process each nucleus individually
    # This prevents large nuclei from dominating the statistic if we just did global masking
    labeled_mask, num_features = label(mask, return_num=True)
    
    if num_features == 0:
        return 0.0

    # Define the boundary region
    # We want the inner rim of the nucleus.
    # Strategy: Mask - Erode(Mask)
    # A radius of 2-3 pixels is typical for capturing the nuclear envelope region in 512x512 images
    erosion_radius = 2
    selem = disk(erosion_radius)
    
    # We can compute the global boundary mask first to speed up extraction
    # However, to be strictly correct per object (avoiding merging boundaries of touching cells),
    # we should ideally do it per object. But given the resolution and typical segmentation,
    # global erosion on the labeled mask or binary mask is a reasonable approximation 
    # if cells are well separated. If cells touch, global erosion might lose the boundary between them.
    # A safer approach for "boundary intensity" is to erode the binary mask.
    
    eroded_mask = binary_erosion(mask, footprint=selem)
    boundary_mask = mask & (~eroded_mask)
    
    # Now we compute the mean intensity of pixels in this boundary region
    # We will compute the mean intensity for each cell's boundary, then average those means.
    # This ensures the feature represents the "average cell state" rather than being weighted by cell size.
    
    boundary_intensities = []
    
    # Iterate through each object
    # Optimization: Instead of looping and masking the whole image, use the labeled mask 
    # and the global boundary mask.
    
    # Get indices where the boundary exists
    boundary_indices = np.where(boundary_mask)
    if len(boundary_indices[0]) == 0:
        return 0.0
        
    boundary_labels = labeled_mask[boundary_indices]
    boundary_values = dapi_channel[boundary_indices]
    
    # We need to group these values by label
    # We can use ndimage.mean to calculate the mean of 'boundary_values' for each label in 'boundary_labels'
    # But ndimage.mean works on the full image array and a label array.
    
    # Let's use ndimage.mean directly on the image with a specific label set
    # We construct a specific label map for boundaries only
    boundary_label_map = np.zeros_like(labeled_mask)
    boundary_label_map[boundary_mask] = labeled_mask[boundary_mask]
    
    # Get unique labels that actually have a boundary (some small spots might disappear after erosion)
    unique_labels = np.unique(boundary_label_map)
    unique_labels = unique_labels[unique_labels > 0] # Remove background
    
    if len(unique_labels) == 0:
        return 0.0
        
    # Calculate mean intensity per object boundary
    # index=unique_labels ensures we only get means for existing objects
    means = ndimage.mean(dapi_channel, labels=boundary_label_map, index=unique_labels)
    
    # The feature is the mean of these means (population average)
    result = np.mean(means)

    return float(result)
