def extract(img, *segmentation_masks):
    # IMPORT ALL REQUIRED PACKAGES AT THE BEGINNING OF THE FUNCTION
    import numpy as np
    from scipy import ndimage
    from skimage import measure, segmentation, graph, filters, morphology

    # Convert to appropriate array type
    arr = np.asarray(img, dtype=np.float32)

    # Handle dimensionality according to dataset format
    # Dataset: (512, 512, 3)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # Standard case
        pass
    elif arr.ndim == 2:
        # Grayscale, treat as single channel
        arr = arr[:, :, np.newaxis]
    else:
        # Unexpected format, return 0.0
        return 0.0

    # --- Step 1: Obtain Instance Segmentation Mask ---
    labels = None

    # Check if valid segmentation masks are provided
    if len(segmentation_masks) > 0:
        # Use the first available mask
        mask_input = segmentation_masks[0]
        
        # Ensure mask matches image spatial dimensions
        if mask_input.shape[:2] == arr.shape[:2]:
            # If mask is already labeled (max > 1), use it directly
            if np.max(mask_input) > 1:
                labels = mask_input.astype(int)
            # If mask is binary (max == 1), label connected components
            elif np.max(mask_input) == 1:
                labels = measure.label(mask_input)
    
    # Fallback: Generate mask from image if no valid segmentation provided
    if labels is None:
        # Use Channel 2 (Blue/DAPI) for nuclei detection as it's the most reliable anchor
        # Channel map: 0=Actin, 1=Tubulin, 2=DAPI
        if arr.shape[2] >= 3:
            dapi_channel = arr[:, :, 2]
        else:
            dapi_channel = np.mean(arr, axis=2) # Fallback for grayscale
            
        # Normalize DAPI channel
        dapi_norm = (dapi_channel - np.min(dapi_channel)) / (np.ptp(dapi_channel) + 1e-6)
        
        # Thresholding (Otsu)
        try:
            thresh = filters.threshold_otsu(dapi_norm)
            binary_mask = dapi_norm > thresh
            
            # Clean up noise
            binary_mask = morphology.remove_small_objects(binary_mask, min_size=20)
            binary_mask = morphology.binary_closing(binary_mask, morphology.disk(2))
            
            # Label instances
            labels = measure.label(binary_mask)
        except Exception:
            # If thresholding fails (e.g., empty image), return 0
            return 0.0

    # If no objects found, return 0
    if labels is None or np.max(labels) == 0:
        return 0.0

    # --- Step 2: Expand Labels to Approximate Cell Boundaries ---
    # Cells might be segmented as nuclei (small, separated). To determine adjacency,
    # we expand labels until they touch or reach a distance limit.
    # Distance=20 pixels is a reasonable approximation for cell radius in 512x512 images of this type.
    try:
        expanded_labels = segmentation.expand_labels(labels, distance=20)
    except AttributeError:
        # Fallback for older skimage versions without expand_labels
        # Use watershed
        distance = ndimage.distance_transform_edt(labels > 0)
        expanded_labels = segmentation.watershed(-distance, labels, mask=labels>0)

    # --- Step 3: Construct Region Adjacency Graph (RAG) ---
    # RAG nodes are regions, edges represent adjacency
    try:
        rag = graph.rag_boundary(expanded_labels, np.ones(expanded_labels.shape), connectivity=2)
    except Exception:
        return 0.0

    # --- Step 4: Calculate Average Degree ---
    neighbor_counts = []
    
    # Iterate over nodes in the graph
    # Note: RAG nodes correspond to label values.
    for node in rag.nodes:
        # Skip background node (usually label 0, but rag_boundary might handle it differently depending on implementation)
        # In skimage RAG, nodes are the label indices.
        if node == 0:
            continue
            
        # Get neighbors for the current node
        neighbors = list(rag.neighbors(node))
        
        # Filter out the background node (0) from the neighbor count
        # We only care about cell-cell adjacency
        cell_neighbors = [n for n in neighbors if n != 0]
        
        neighbor_counts.append(len(cell_neighbors))

    # --- Step 5: Aggregate Results ---
    if not neighbor_counts:
        return 0.0
        
    result = np.mean(neighbor_counts)

    return float(result)
