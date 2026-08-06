def extract(img, seg):
    import numpy as np
    from scipy import ndimage

    arr = np.asarray(img)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        return float(0.0)

    arr = arr.astype(np.float32, copy=False)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return float(0.0)

    finite_median = float(np.median(arr[finite]))
    arr = np.where(finite, arr, finite_median).astype(np.float32, copy=False)

    if not seg or not isinstance(seg, dict):
        return float(0.0)

    def _mask_from_seg(key):
        m = seg.get(key)
        if m is None:
            return None
        m = np.squeeze(np.asarray(m))
        if m.shape != arr.shape:
            return None
        return m > 0

    cell = _mask_from_seg("mask_cell")
    if cell is None or int(np.count_nonzero(cell)) < 100:
        return float(0.0)

    nucleus = _mask_from_seg("mask_nucleus")
    if nucleus is not None:
        nucleus = nucleus & cell
        roi = cell & (~nucleus)
    else:
        roi = cell.copy()

    if int(np.count_nonzero(roi)) < 100:
        return float(0.0)

    outside_cell = (~cell) & np.isfinite(arr)
    if np.count_nonzero(outside_cell) >= 100:
        background = float(np.median(arr[outside_cell]))
    else:
        background = float(np.percentile(arr[np.isfinite(arr)], 1.0))

    img_corr = arr - background
    img_corr[img_corr < 0] = 0.0

    if nucleus is not None and np.count_nonzero(nucleus) > 0:
        dist = ndimage.distance_transform_edt(~nucleus)
    else:
        ys, xs = np.nonzero(cell)
        if ys.size == 0:
            return float(0.0)
        cy = float(np.mean(ys))
        cx = float(np.mean(xs))
        yy, xx = np.indices(arr.shape)
        dist = np.sqrt((yy.astype(np.float32) - cy) ** 2 + (xx.astype(np.float32) - cx) ** 2)

    roi_dist = dist[roi]
    if roi_dist.size < 100 or not np.all(np.isfinite(roi_dist)):
        return float(0.0)

    q33, q67 = np.percentile(roi_dist, [33.333333, 66.666667])
    if not np.isfinite(q33) or not np.isfinite(q67) or q67 <= q33:
        return float(0.0)

    proximal = roi & (dist <= q33)
    distal = roi & (dist >= q67)

    min_pixels = max(25, int(0.01 * np.count_nonzero(roi)))
    if np.count_nonzero(proximal) < min_pixels or np.count_nonzero(distal) < min_pixels:
        return float(0.0)

    proximal_values = img_corr[proximal]
    distal_values = img_corr[distal]

    if proximal_values.size == 0 or distal_values.size == 0:
        return float(0.0)

    proximal_mean = float(np.mean(proximal_values))
    distal_mean = float(np.mean(distal_values))

    eps = 1e-6
    if not np.isfinite(proximal_mean) or not np.isfinite(distal_mean) or proximal_mean <= eps:
        return float(0.0)

    result = distal_mean / proximal_mean
    if not np.isfinite(result):
        return float(0.0)

    return float(result)
