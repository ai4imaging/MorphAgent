def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    
    # Convert to float32 for calculation precision
    arr = np.asarray(img, dtype=np.float32)
    
    # 1. Validate Input Dimensions
    # Dataset spec: (512, 512, 3). If 2D (512, 512), it might be a single channel or projection.
    # We need the Tubulin channel (Index 1).
    if arr.ndim == 3 and arr.shape[2] >= 2:
        # Standard case: (H, W, C)
        tubulin_channel = arr[..., 1]
        # We might use Actin (0) or DAPI (2) for fallback masking
        actin_channel = arr[..., 0]
        dapi_channel = arr[..., 2] if arr.shape[2] > 2 else None
    elif arr.ndim == 2:
        # Fallback: Treat the single channel as the target if that's all we have,
        # though this is unlikely given the dataset description.
        tubulin_channel = arr
        actin_channel = arr
        dapi_channel = None
    else:
        return 0.0

    # 2. Define the Region of Interest (Cytoplasm)
    # The goal is to measure Tubulin intensity specifically in the cytoplasm.
    
    mask_roi = None

    # Case A: Segmentation Masks Provided
    if len(segmentation_masks) > 0:
        # We expect masks to be passed. Common scenarios:
        # 1 mask: Likely cells or nuclei.
        # 2 masks: Likely nuclei and cells (or cytoplasm).
        
        # Helper to ensure mask is boolean
        def to_bool(m):
            return np.asarray(m) > 0

        if len(segmentation_masks) >= 2:
            # Heuristic: Nuclei masks usually have smaller total area than cell masks
            m1 = segmentation_masks[0]
            m2 = segmentation_masks[1]
            
            area1 = np.sum(m1 > 0)
            area2 = np.sum(m2 > 0)
            
            # Assume smaller area is nuclei, larger is whole cell
            if area1 < area2:
                nuclei_mask = to_bool(m1)
                cell_mask = to_bool(m2)
            else:
                nuclei_mask = to_bool(m2)
                cell_mask = to_bool(m1)
                
            # Cytoplasm = Cell - Nuclei
            mask_roi = np.logical_and(cell_mask, np.logical_not(nuclei_mask))
            
        elif len(segmentation_masks) == 1:
            # Only one mask. If it's a cell mask, use it directly.
            # If it's a nuclei mask, we can't get cytoplasm easily, but using the mask 
            # is better than nothing, or we might invert it if it covers the whole image?
            # Safest bet: Treat it as the region of interest.
            mask_roi = to_bool(segmentation_masks[0])

    # Case B: No Segmentation Masks (Fallback)
    # We must generate a mask from the image itself to avoid averaging background zeros.
    if mask_roi is None:
        # Create a "Cell Body" mask using Actin (Ch0) and Tubulin (Ch1).
        # Both are cytoskeletal and define the cell shape well.
        # We use a low threshold to separate foreground from background.
        
        # Combine channels for robust detection
        structure_signal = tubulin_channel + actin_channel
        
        # Simple background estimation:
        # The background is usually the mode of the histogram, or very low values.
        # We use a statistical threshold: mean + 0.5 * std (conservative) or a fixed value.
        # Given uint8 data, background is often < 20.
        threshold = np.mean(structure_signal) * 0.8  # Adaptive threshold
        
        # Create binary mask
        foreground_mask = structure_signal > threshold
        
        # If we have DAPI, we can try to exclude the nucleus for a "cytoplasm" approximation
        if dapi_channel is not None:
            # Nuclei are usually very bright in DAPI
            nuclei_threshold = np.percentile(dapi_channel, 95)
            # Ensure threshold is reasonable (not just noise)
            nuclei_threshold = max(nuclei_threshold, 30.0) 
            nuclei_mask = dapi_channel > nuclei_threshold
            
            # Cytoplasm approx: Foreground AND NOT Nuclei
            mask_roi = np.logical_and(foreground_mask, np.logical_not(nuclei_mask))
        else:
            mask_roi = foreground_mask

    # 3. Calculate Feature
    # Extract pixels within the ROI
    if mask_roi is not None and np.any(mask_roi):
        pixels = tubulin_channel[mask_roi]
        
        # Calculate mean intensity
        # We return the raw intensity value (0-255 scale) as it's physically meaningful
        # for comparison across images in the same dataset.
        mean_intensity = np.mean(pixels)
        return float(mean_intensity)
    else:
        # If mask is empty (no cells found), return 0.0
        return 0.0
