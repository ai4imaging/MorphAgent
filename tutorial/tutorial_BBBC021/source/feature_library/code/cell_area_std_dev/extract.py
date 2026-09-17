def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage.measure import label, regionprops
    from skimage.filters import threshold_otsu, threshold_triangle, sobel
    from skimage.segmentation import watershed
    from skimage.morphology import binary_opening, disk

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality
    # Expected shape: (512, 512, 3)
    if arr.ndim != 3 or arr.shape[2] != 3:
        # Fallback for unexpected shapes, though dataset spec says (512, 512, 3)
        return 0.0

    # Check if a valid segmentation mask is provided
    labeled_cells = None
    
    # If masks are provided, try to use the first one (assuming it's a cell/nuclei mask)
    if len(segmentation_masks) > 0 and segmentation_masks[0] is not None:
        mask_candidate = segmentation_masks[0]
        # Ensure mask matches image spatial dimensions
        if mask_candidate.shape[:2] == arr.shape[:2]:
            labeled_cells = mask_candidate.astype(int)

    # If no valid mask provided, perform on-the-fly segmentation
    if labeled_cells is None:
        # 1. Normalize image channels
        # Channel 0: Actin (Red), Channel 1: Tubulin (Green), Channel 2: DAPI (Blue)
        # Normalize to [0, 1]
        arr_norm = arr / 255.0
        
        ch_actin = arr_norm[..., 0]
        ch_tubulin = arr_norm[..., 1]
        ch_dapi = arr_norm[..., 2]

        # 2. Detect Nuclei (Seeds)
        # Use DAPI channel. Smooth slightly to reduce noise.
        dapi_smooth = ndimage.gaussian_filter(ch_dapi, sigma=2)
        try:
            thresh_nuc = threshold_otsu(dapi_smooth)
            mask_nuc = dapi_smooth > thresh_nuc
            # Clean up small noise
            mask_nuc = binary_opening(mask_nuc, footprint=disk(2))
            # Label nuclei
            markers, _ = label(mask_nuc, return_num=True)
        except Exception:
            # Fallback if thresholding fails (e.g. empty image)
            return 0.0

        # 3. Define Cell Body (Mask)
        # Combine Actin and Tubulin for robust cell shape
        # Use maximum projection to capture full extent
        cell_signal = np.maximum(ch_actin, ch_tubulin)
        cell_smooth = ndimage.gaussian_filter(cell_signal, sigma=2)
        
        try:
            # Triangle threshold is often better for cytoplasm which has a long tail in histogram
            thresh_cell = threshold_triangle(cell_smooth)
            mask_cell = cell_smooth > thresh_cell
        except Exception:
            # Fallback to Otsu if Triangle fails
            try:
                thresh_cell = threshold_otsu(cell_smooth)
                mask_cell = cell_smooth > thresh_cell
            except Exception:
                return 0.0

        # 4. Watershed Segmentation
        # Use gradient of cell signal as elevation map
        elevation_map = sobel(cell_smooth)
        
        # Run watershed
        # markers: labeled nuclei
        # mask: binary cell body mask
        labeled_cells = watershed(elevation_map, markers, mask=mask_cell)

    # Calculate Feature: Standard Deviation of Cell Areas
    
    # Extract properties
    props = regionprops(labeled_cells)
    
    # Collect areas
    areas = []
    for prop in props:
        # Filter out very small objects (debris/noise)
        # A typical MCF-7 nucleus is ~100-200 pixels, cell is larger.
        # Threshold at 50 pixels to be safe.
        if prop.area > 50:
            areas.append(prop.area)
            
    areas = np.array(areas, dtype=np.float64)

    # Compute Standard Deviation
    if areas.size < 2:
        # Std dev is undefined or 0 for 0 or 1 cell
        return 0.0
    
    # ddof=1 for sample standard deviation
    std_dev_area = np.std(areas, ddof=1)

    return float(std_dev_area)
