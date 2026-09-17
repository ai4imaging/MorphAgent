def extract(img, seg):
    import numpy as np
    from scipy import ndimage

    arr = np.asarray(img)
    arr = np.squeeze(arr)

    if arr.ndim == 2:
        img2d = arr.astype(np.float64, copy=False)
    elif arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            img2d = np.mean(arr[..., :3].astype(np.float64, copy=False), axis=-1)
        elif arr.shape[0] in (3, 4):
            img2d = np.mean(arr[:3, ...].astype(np.float64, copy=False), axis=0)
        else:
            img2d = np.max(arr.astype(np.float64, copy=False), axis=0)
    else:
        return 0.0

    if img2d.ndim != 2 or img2d.size == 0:
        return 0.0

    if not seg or not isinstance(seg, dict):
        return 0.0

    def _mask_bool(mask):
        if mask is None:
            return None
        m = np.asarray(mask)
        m = np.squeeze(m)
        if m.ndim == 3:
            if m.shape[-1] in (3, 4):
                m = np.max(m[..., :3], axis=-1)
            elif m.shape[0] in (3, 4):
                m = np.max(m[:3, ...], axis=0)
            else:
                m = np.max(m, axis=0)
        if m.shape != img2d.shape:
            return None
        return np.isfinite(m) & (m > 0)

    cell = _mask_bool(seg.get("mask_cell"))
    if cell is None or int(np.count_nonzero(cell)) == 0:
        return 0.0

    nucleus = _mask_bool(seg.get("mask_nucleus"))
    bundle = _mask_bool(seg.get("mask_bundle"))
    filament = _mask_bool(seg.get("mask_filament"))

    if nucleus is None:
        nucleus = np.zeros_like(cell, dtype=bool)

    axon_raw = np.zeros_like(cell, dtype=bool)
    if bundle is not None:
        axon_raw |= bundle
    if filament is not None:
        axon_raw |= filament

    axon_raw &= ~nucleus
    min_pixels = 25

    if np.count_nonzero(axon_raw) >= min_pixels:
        axon_in_cell = axon_raw & cell
        if np.count_nonzero(axon_in_cell) >= min_pixels:
            axon_mask = axon_in_cell
        else:
            axon_mask = axon_raw
        soma_mask = cell & (~nucleus) & (~axon_mask)
        if np.count_nonzero(soma_mask) < min_pixels:
            soma_mask = cell & (~nucleus)
    else:
        cytoplasm = cell & (~nucleus)
        if np.count_nonzero(cytoplasm) < 2 * min_pixels or np.count_nonzero(nucleus) == 0:
            return 0.0

        dist_to_nucleus = ndimage.distance_transform_edt(~nucleus)
        vals = dist_to_nucleus[cytoplasm]
        vals = vals[np.isfinite(vals)]
        if vals.size < 2 * min_pixels:
            return 0.0

        thresh = np.percentile(vals, 35.0)
        soma_mask = cytoplasm & (dist_to_nucleus <= thresh)
        axon_mask = cytoplasm & (dist_to_nucleus > thresh)

    if np.count_nonzero(axon_mask) < min_pixels or np.count_nonzero(soma_mask) < min_pixels:
        return 0.0

    finite_img = np.isfinite(img2d)
    if not np.any(finite_img):
        return 0.0

    outside_cell = (~cell) & finite_img
    if np.count_nonzero(outside_cell) >= min_pixels:
        background = np.percentile(img2d[outside_cell], 10.0)
    else:
        background = np.percentile(img2d[finite_img], 1.0)

    corrected = img2d - background
    corrected[~finite_img] = np.nan
    corrected = np.maximum(corrected, 0.0)

    axon_vals = corrected[axon_mask & finite_img]
    soma_vals = corrected[soma_mask & finite_img]

    if axon_vals.size < min_pixels or soma_vals.size < min_pixels:
        return 0.0

    axon_mean = np.nanmean(axon_vals)
    soma_mean = np.nanmean(soma_vals)

    if not np.isfinite(axon_mean) or not np.isfinite(soma_mean) or soma_mean <= 0:
        return 0.0

    result = axon_mean / soma_mean
    if not np.isfinite(result):
        return 0.0

    return float(result)
