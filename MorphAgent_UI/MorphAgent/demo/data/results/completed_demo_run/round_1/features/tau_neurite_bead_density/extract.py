def extract(img, seg):
    import numpy as np
    from scipy import ndimage
    from skimage.morphology import skeletonize, remove_small_objects
    from skimage.filters import threshold_otsu
    from skimage.feature import peak_local_max

    arr = np.asarray(img, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2 or arr.size == 0:
        return float(0.0)

    finite = np.isfinite(arr)
    if not np.any(finite):
        return float(0.0)
    max_finite = float(np.max(arr[finite]))
    arr = np.nan_to_num(arr, nan=0.0, posinf=max_finite, neginf=0.0).astype(np.float32, copy=False)

    shape = arr.shape

    def _as_bool_mask(mask):
        if mask is None:
            return None
        m = np.asarray(mask)
        m = np.squeeze(m)
        if m.shape != shape:
            return None
        return m > 0

    seg_dict = seg if isinstance(seg, dict) else {}

    cell_mask = _as_bool_mask(seg_dict.get("mask_cell"))
    if cell_mask is None or not np.any(cell_mask):
        cell_mask = np.ones(shape, dtype=bool)

    nucleus_mask = _as_bool_mask(seg_dict.get("mask_nucleus"))
    if nucleus_mask is not None and np.any(nucleus_mask):
        cytoplasm_mask = cell_mask & (~nucleus_mask)
    else:
        cytoplasm_mask = cell_mask.copy()

    if not np.any(cytoplasm_mask):
        return float(0.0)

    norm_values = arr[cytoplasm_mask]
    if norm_values.size < 2:
        return float(0.0)

    p_low = float(np.percentile(norm_values, 1.0))
    p_high = float(np.percentile(norm_values, 99.5))
    if not np.isfinite(p_low) or not np.isfinite(p_high) or p_high <= p_low:
        return float(0.0)

    img_norm = np.clip((arr - p_low) / (p_high - p_low), 0.0, 1.0).astype(np.float32, copy=False)
    img_smooth = ndimage.gaussian_filter(img_norm, sigma=1.0)

    bundle_mask = _as_bool_mask(seg_dict.get("mask_bundle"))
    filament_mask = _as_bool_mask(seg_dict.get("mask_filament"))

    neurite_mask = np.zeros(shape, dtype=bool)
    if bundle_mask is not None:
        neurite_mask |= bundle_mask
    if filament_mask is not None:
        neurite_mask |= filament_mask

    neurite_mask &= cytoplasm_mask

    if np.any(neurite_mask):
        neurite_mask = ndimage.binary_closing(neurite_mask, structure=np.ones((3, 3), dtype=bool))
        neurite_mask = remove_small_objects(neurite_mask.astype(bool), min_size=20)

    if np.count_nonzero(neurite_mask) < 20:
        vals = img_smooth[cytoplasm_mask]
        if vals.size < 10:
            return float(0.0)
        p75 = float(np.percentile(vals, 75.0))
        try:
            otsu_thr = float(threshold_otsu(vals))
        except Exception:
            otsu_thr = p75
        fallback_thr = max(otsu_thr, p75)
        neurite_mask = (img_smooth > fallback_thr) & cytoplasm_mask
        neurite_mask = remove_small_objects(neurite_mask.astype(bool), min_size=30)
        if np.count_nonzero(neurite_mask) < 20:
            return float(0.0)

    skeleton = skeletonize(neurite_mask.astype(bool))
    skeleton_length = int(np.count_nonzero(skeleton))
    if skeleton_length < 10:
        return float(0.0)

    distance_to_skeleton = ndimage.distance_transform_edt(~skeleton)
    alignment_radius = 4.0
    near_skeleton = distance_to_skeleton <= alignment_radius
    peak_search_mask = cytoplasm_mask & near_skeleton

    if not np.any(peak_search_mask):
        return float(0.0)

    vals = img_smooth[cytoplasm_mask]
    med = float(np.median(vals))
    mad_sigma = float(1.4826 * np.median(np.abs(vals - med)))
    p90 = float(np.percentile(vals, 90.0))
    threshold_abs = max(p90, med + 2.0 * mad_sigma)

    if not np.isfinite(threshold_abs):
        return float(0.0)

    coords = peak_local_max(
        img_smooth,
        min_distance=3,
        threshold_abs=threshold_abs,
        exclude_border=False,
        labels=peak_search_mask.astype(np.uint8),
    )

    if coords is None or len(coords) == 0:
        return float(0.0)

    aligned_peak_count = int(np.count_nonzero(distance_to_skeleton[coords[:, 0], coords[:, 1]] <= alignment_radius))
    density = 100.0 * float(aligned_peak_count) / float(skeleton_length)

    if not np.isfinite(density):
        return float(0.0)
    return float(density)
