"""Shared mitochondrial feature implementations for the locked HSC 25-feature vocabulary.

Each function matches Supplementary feature list 2. Inputs are parsed from
`extract(img, *segmentation_masks)`:

  img                 2D mito intensity, or (H, W, C) with mito in channel 0
  segmentation_masks  optional mito instance labels, cell mask, nucleus mask
                      (order does not matter; integer labels vs binary is inferred)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, stats
from scipy.spatial import cKDTree
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize


def _as2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        # (H, W, C) or (C, H, W). Prefer last-axis channels if small.
        if arr.shape[-1] <= 4:
            return np.asarray(arr[..., 0], dtype=np.float64)
        return np.asarray(arr[0], dtype=np.float64)
    if arr.ndim == 2:
        return np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        raise ValueError("expected a 2D image")
    # 3D stack → MIP
    return np.asarray(arr.max(axis=0), dtype=np.float64)


def parse_inputs(img, segmentation_masks):
    mito = _as2d(img)
    h, w = mito.shape
    labels = None
    cell = None
    nucleus = None

    parsed = []
    for mask in segmentation_masks:
        if mask is None:
            continue
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[..., 0] if m.shape[-1] <= 4 else m[0]
        if m.shape[:2] != (h, w):
            continue
        parsed.append(m)

    # Heuristic: the integer label map with the most unique values is mitochondria.
    int_maps = [m for m in parsed if np.issubdtype(m.dtype, np.integer) and int(m.max()) > 1]
    if int_maps:
        labels = max(int_maps, key=lambda m: len(np.unique(m)))
    binaries = []
    for m in parsed:
        if labels is not None and m is labels:
            continue
        binaries.append(m > 0)
    if binaries:
        # largest foreground → cell; smaller compact → nucleus if two masks
        binaries_sorted = sorted(binaries, key=lambda b: int(b.sum()), reverse=True)
        cell = binaries_sorted[0]
        if len(binaries_sorted) >= 2:
            nucleus = binaries_sorted[-1]

    if labels is None:
        fg = mito > np.percentile(mito, 80) if mito.size else np.zeros_like(mito, dtype=bool)
        if cell is not None:
            fg = fg & cell
        labels = label(fg)

    if cell is None:
        cell = labels > 0
        if cell.sum() == 0:
            cell = mito > 0

    return mito, labels.astype(np.int32), cell.astype(bool), None if nucleus is None else nucleus.astype(bool)


def _fail() -> float:
    return 0.0


def _safe(fn):
    def wrapped(img, *segmentation_masks):
        try:
            mito, labels, cell, nucleus = parse_inputs(img, segmentation_masks)
            val = fn(mito, labels, cell, nucleus)
            if val is None or not np.isfinite(val):
                return _fail()
            return float(val)
        except Exception:
            return _fail()

    wrapped.__name__ = fn.__name__
    return wrapped


def _object_props(labels):
    return [p for p in regionprops(labels) if p.area >= 4]


# ---------------------------------------------------------------------------
# 24 code features (Supplementary feature list 2)
# ---------------------------------------------------------------------------

@_safe
def mitochondria_background_signal_heterogeneity(mito, labels, cell, nucleus):
    """Std of intensities in the background (outside mitochondrial labels)."""
    bg = cell & (labels == 0)
    if bg.sum() < 16:
        bg = labels == 0
    pix = mito[bg]
    if pix.size < 8:
        return 0.0
    return float(np.std(pix))


@_safe
def mito_perimeter_intensity_gradient_mean(mito, labels, cell, nucleus):
    """Mean inward-minus-outward intensity difference along object perimeters."""
    props = _object_props(labels)
    if not props:
        return 0.0
    scores, weights = [], []
    for p in props:
        minr, minc, maxr, maxc = p.bbox
        sl = (slice(minr, maxr), slice(minc, maxc))
        obj = (labels[sl] == p.label)
        if obj.sum() < 8:
            continue
        eroded = ndimage.binary_erosion(obj)
        dilated = ndimage.binary_dilation(obj)
        inside_ring = obj & ~eroded
        outside_ring = dilated & ~obj
        if inside_ring.sum() == 0 or outside_ring.sum() == 0:
            continue
        patch = mito[sl]
        delta = patch[inside_ring].mean() - patch[outside_ring].mean()
        scores.append(delta)
        weights.append(float(inside_ring.sum()))
    if not scores:
        return 0.0
    return float(np.average(scores, weights=weights))


@_safe
def background_mito_signal_to_noise(mito, labels, cell, nucleus):
    """Mean mito intensity / std of a background ring around the cell."""
    mito_pix = mito[labels > 0]
    if mito_pix.size == 0:
        return 0.0
    dilated = ndimage.binary_dilation(cell, iterations=20)
    ring = dilated & ~cell
    if ring.sum() < 16:
        ring = ~cell
    noise = mito[ring]
    sd = float(np.std(noise)) if noise.size else 0.0
    if sd <= 1e-12:
        return 0.0
    return float(np.mean(mito_pix) / sd)


@_safe
def mitochondria_neighbor_intensity_autocorrelation(mito, labels, cell, nucleus):
    """Local Moran's I on mitochondrial intensities inside the cell mask."""
    mask = cell.copy()
    if mask.sum() < 32:
        return 0.0
    x = mito[mask].astype(np.float64)
    x = x - x.mean()
    if np.allclose(x, 0):
        return 0.0
    # 8-connected lag-1 via convolution on the full image, then restrict to mask
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float64)
    centered = np.zeros_like(mito, dtype=np.float64)
    centered[mask] = x
    neigh = ndimage.convolve(centered, kernel, mode="constant")
    count = ndimage.convolve(mask.astype(np.float64), kernel, mode="constant")
    valid = mask & (count > 0)
    if valid.sum() < 16:
        return 0.0
    wxi = neigh[valid]
    xi = centered[valid]
    num = np.sum(xi * wxi)
    den = np.sum(xi * xi)
    if den <= 1e-12:
        return 0.0
    # scale by mean neighbour count
    wbar = count[valid].mean()
    return float(num / (den * max(wbar, 1.0)))


@_safe
def mitochondria_texture_anisotropy_glcm(mito, labels, cell, nucleus):
    """CV of GLCM contrast across 0/45/90/135 degrees."""
    region = mito.copy()
    m = cell & (labels > 0)
    if m.sum() < 64:
        m = cell
    if m.sum() < 64:
        return 0.0
    lo, hi = np.percentile(region[m], (1, 99))
    if hi <= lo:
        return 0.0
    q = np.clip((region - lo) / (hi - lo), 0, 1)
    q[~m] = 0
    q8 = (q * 15).astype(np.uint8)
    glcm = graycomatrix(
        q8,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=16,
        symmetric=True,
        normed=True,
    )
    contrast = graycoprops(glcm, "contrast").ravel()
    mu = contrast.mean()
    if mu <= 1e-12:
        return 0.0
    return float(contrast.std() / mu)


@_safe
def mitochondria_long_tubule_fraction(mito, labels, cell, nucleus):
    """Fraction of objects whose major-axis length exceeds a high-length threshold."""
    props = _object_props(labels)
    if not props:
        return 0.0
    lengths = np.array([p.major_axis_length for p in props], dtype=np.float64)
    # high-length = 75th percentile of major axis
    thr = np.percentile(lengths, 75)
    if thr <= 0:
        return 0.0
    return float(np.mean(lengths > thr))


@_safe
def mitochondrial_area_fraction(mito, labels, cell, nucleus):
    """Fraction of the image occupied by mitochondrial labels."""
    return float((labels > 0).mean())


@_safe
def mitochondria_local_autocorrelation_length(mito, labels, cell, nucleus):
    """Radius at which 2D autocorrelation of mito intensity decays to 1/e."""
    mask = cell
    if mask.sum() < 64:
        return 0.0
    x = mito.copy()
    x = x - np.mean(x[mask])
    x[~mask] = 0
    # FFT autocorrelation, radial profile
    f = np.fft.rfft2(x)
    ac = np.fft.irfft2(np.abs(f) ** 2, s=x.shape).real
    ac = np.fft.fftshift(ac)
    ac /= ac.max() if ac.max() != 0 else 1.0
    cy, cx = np.array(ac.shape) // 2
    yy, xx = np.ogrid[: ac.shape[0], : ac.shape[1]]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = int(min(cy, cx) * 0.5)
    target = 1.0 / np.e
    for radius in range(1, max(rmax, 2)):
        ring = (r >= radius - 0.5) & (r < radius + 0.5)
        if ring.sum() == 0:
            continue
        if ac[ring].mean() <= target:
            return float(radius)
    return float(rmax)


@_safe
def mito_intensity_autocorrelation_length(mito, labels, cell, nucleus):
    """Autocorrelation length restricted to mitochondrial pixels."""
    masked = np.zeros_like(mito)
    m = labels > 0
    if m.sum() < 64:
        return 0.0
    masked[m] = mito[m] - mito[m].mean()
    f = np.fft.rfft2(masked)
    ac = np.fft.irfft2(np.abs(f) ** 2, s=masked.shape).real
    ac = np.fft.fftshift(ac)
    peak = ac.max()
    if peak == 0:
        return 0.0
    ac /= peak
    cy, cx = np.array(ac.shape) // 2
    yy, xx = np.ogrid[: ac.shape[0], : ac.shape[1]]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = int(min(cy, cx) * 0.5)
    target = 1.0 / np.e
    for radius in range(1, max(rmax, 2)):
        ring = (r >= radius - 0.5) & (r < radius + 0.5)
        if ring.sum() and ac[ring].mean() <= target:
            return float(radius)
    return float(rmax)


@_safe
def mitochondria_to_cell_intensity_dynamic_range_ratio(mito, labels, cell, nucleus):
    """(p95-p5) inside mitochondria / (p95-p5) inside the cell."""
    mito_pix = mito[labels > 0]
    cell_pix = mito[cell]
    if mito_pix.size < 16 or cell_pix.size < 16:
        return 0.0
    dr_m = np.percentile(mito_pix, 95) - np.percentile(mito_pix, 5)
    dr_c = np.percentile(cell_pix, 95) - np.percentile(cell_pix, 5)
    if dr_c <= 1e-12:
        return 0.0
    return float(dr_m / dr_c)


@_safe
def background_noise_robust_snr(mito, labels, cell, nucleus):
    """median(mito) / MAD(background)."""
    sig = mito[labels > 0]
    bg = mito[~cell]
    if sig.size < 8 or bg.size < 8:
        bg = mito[labels == 0]
    if sig.size < 8 or bg.size < 8:
        return 0.0
    mad = np.median(np.abs(bg - np.median(bg))) * 1.4826
    if mad <= 1e-12:
        return 0.0
    return float(np.median(sig) / mad)


@_safe
def mito_intensity_geodesic_variation(mito, labels, cell, nucleus):
    """Slope of intensity CV vs geodesic distance from the brightest pixel."""
    m = labels > 0
    if m.sum() < 32:
        return 0.0
    # geodesic distance from the global brightest mito pixel along the mask
    idx = np.argmax(np.where(m, mito, -np.inf))
    seed = np.zeros_like(m, dtype=bool)
    seed.flat[idx] = True
    # distance transform on inverted mask (inf outside)
    inv = np.where(m, 0.0, 1e6)
    # multi-source: set seed to 0
    dist = ndimage.distance_transform_edt(~seed)
    dist = np.where(m, dist, np.nan)
    vals = mito[m]
    d = dist[m]
    order = np.argsort(d)
    d, vals = d[order], vals[order]
    n_bins = 8
    if vals.size < n_bins * 4:
        return 0.0
    edges = np.linspace(d.min(), d.max() + 1e-6, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        sel = (d >= edges[i]) & (d < edges[i + 1])
        if sel.sum() < 4:
            continue
        mu = vals[sel].mean()
        if mu <= 1e-12:
            continue
        xs.append(0.5 * (edges[i] + edges[i + 1]))
        ys.append(vals[sel].std() / mu)
    if len(xs) < 3:
        return 0.0
    slope, _, _, _, _ = stats.linregress(xs, ys)
    return float(slope)


@_safe
def mitochondria_local_entropy_mean(mito, labels, cell, nucleus):
    """Mean local Shannon entropy of mito intensity inside the cell."""
    from skimage.filters.rank import entropy
    from skimage.morphology import disk as skdisk

    m = cell
    if m.sum() < 64:
        return 0.0
    lo, hi = np.percentile(mito[m], (1, 99))
    if hi <= lo:
        return 0.0
    q = np.clip((mito - lo) / (hi - lo), 0, 1)
    q8 = (q * 255).astype(np.uint8)
    ent = entropy(q8, skdisk(3))
    return float(ent[m].mean())


@_safe
def mitochondrial_radial_anisotropy_index(mito, labels, cell, nucleus):
    """CV of sector-wise radial mito intensity profiles around the cell centroid."""
    ys, xs = np.nonzero(cell)
    if ys.size < 32:
        return 0.0
    cy, cx = ys.mean(), xs.mean()
    yy, xx = np.ogrid[: mito.shape[0], : mito.shape[1]]
    ang = np.arctan2(yy - cy, xx - cx)
    rad = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    n_sec = 8
    n_r = 6
    rmax = rad[cell].max()
    if rmax <= 1:
        return 0.0
    profiles = []
    for s in range(n_sec):
        a0, a1 = -np.pi + s * (2 * np.pi / n_sec), -np.pi + (s + 1) * (2 * np.pi / n_sec)
        sector = cell & (ang >= a0) & (ang < a1)
        row = []
        for b in range(n_r):
            r0, r1 = b * rmax / n_r, (b + 1) * rmax / n_r
            ring = sector & (rad >= r0) & (rad < r1)
            row.append(mito[ring].mean() if ring.sum() else 0.0)
        profiles.append(row)
    profiles = np.asarray(profiles)
    mu = profiles.mean(axis=0)
    # CV across sectors of the radial profile L2
    norms = np.linalg.norm(profiles, axis=1)
    if norms.mean() <= 1e-12:
        return 0.0
    return float(norms.std() / norms.mean())


@_safe
def mitochondria_network_continuity_score(mito, labels, cell, nucleus):
    """Network-like vs fragmented: skeleton length / sqrt(area) averaged."""
    props = _object_props(labels)
    if not props:
        return 0.0
    scores = []
    for p in props:
        minr, minc, maxr, maxc = p.bbox
        obj = labels[minr:maxr, minc:maxc] == p.label
        skel = skeletonize(obj)
        length = float(skel.sum())
        area = float(obj.sum())
        scores.append(length / np.sqrt(area) if area > 0 else 0.0)
    return float(np.mean(scores))


@_safe
def mitochondria_length_width_skewness(mito, labels, cell, nucleus):
    """Skewness of per-object major/minor axis ratio."""
    props = _object_props(labels)
    if len(props) < 4:
        return 0.0
    ratios = []
    for p in props:
        minor = p.minor_axis_length if p.minor_axis_length > 1e-6 else 1e-6
        ratios.append(p.major_axis_length / minor)
    if np.allclose(ratios, ratios[0]):
        return 0.0
    g1 = stats.skew(ratios, bias=False)
    return float(0.0 if not np.isfinite(g1) else g1)


@_safe
def mitochondria_intensity_entropy(mito, labels, cell, nucleus):
    """Shannon entropy of the mito intensity histogram inside the cell."""
    pix = mito[cell]
    if pix.size < 16:
        return 0.0
    hist, _ = np.histogram(pix, bins=64, density=True)
    hist = hist[hist > 0]
    hist = hist / hist.sum()
    return float(-(hist * np.log2(hist)).sum())


@_safe
def mito_clustered_vs_isolated_object_fraction(mito, labels, cell, nucleus):
    """Fraction of mitochondrial area in objects that have a nearby neighbour."""
    props = _object_props(labels)
    if len(props) < 2:
        return 0.0
    cents = np.array([p.centroid for p in props])
    areas = np.array([p.area for p in props], dtype=np.float64)
    tree = cKDTree(cents)
    # small distance: 8 pixels after a slight dilation-equivalent
    d, _ = tree.query(cents, k=2)
    nn = d[:, 1]
    clustered = nn <= 8.0
    if areas.sum() == 0:
        return 0.0
    return float(areas[clustered].sum() / areas.sum())


@_safe
def mitochondria_intensity_gini_inside_cell(mito, labels, cell, nucleus):
    """Gini coefficient of mito intensities inside the cell mask."""
    x = np.sort(mito[cell].ravel())
    x = x[np.isfinite(x)]
    if x.size < 8:
        return 0.0
    if np.allclose(x, 0):
        return 0.0
    n = x.size
    # Gini = 2*sum(i*x_i)/(n*sum x) - (n+1)/n
    g = 2.0 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n
    return float(np.clip(g, 0.0, 1.0))


@_safe
def mitochondria_area_fraction_in_perinuclear_band(mito, labels, cell, nucleus):
    """Fraction of mito area in a perinuclear band (2–5 px if no scale; else ~2–5 µm)."""
    if nucleus is None or nucleus.sum() < 8:
        # approximate nucleus as the inner 30% radius of the cell
        ys, xs = np.nonzero(cell)
        if ys.size == 0:
            return 0.0
        cy, cx = ys.mean(), xs.mean()
        yy, xx = np.ogrid[: mito.shape[0], : mito.shape[1]]
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        rmax = r[cell].max()
        nucleus = cell & (r <= 0.25 * rmax)
    band = ndimage.binary_dilation(nucleus, iterations=12) & ~nucleus
    mito_area = labels > 0
    tot = mito_area.sum()
    if tot == 0:
        return 0.0
    return float((mito_area & band).sum() / tot)


@_safe
def mitochondria_intensity_entropy_within_cell(mito, labels, cell, nucleus):
    """Shannon entropy of intensities inside the mitochondrial region (64 bins)."""
    pix = mito[labels > 0]
    if pix.size < 16:
        return 0.0
    hist, _ = np.histogram(pix, bins=64, density=False)
    p = hist.astype(np.float64)
    p = p[p > 0]
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


@_safe
def mitochondrial_clusteredness_by_nearest_neighbor(mito, labels, cell, nucleus):
    """Mean NN distance vs a uniform spatial null (Clark–Evans style)."""
    props = _object_props(labels)
    if len(props) < 3:
        return 0.0
    cents = np.array([p.centroid for p in props])
    tree = cKDTree(cents)
    nn = tree.query(cents, k=2)[0][:, 1]
    obs = float(nn.mean())
    area = float(cell.sum()) if cell.sum() else float(mito.size)
    n = len(props)
    # expected NN under CSR: 0.5 / sqrt(density)
    exp = 0.5 / np.sqrt(n / area) if n > 0 else np.nan
    if not np.isfinite(exp) or exp <= 1e-12:
        return 0.0
    # higher clusteredness → smaller obs/exp → invert so high = clustered
    return float(max(0.0, 1.0 - obs / exp))


@_safe
def mitochondrial_network_fragmentation_score(mito, labels, cell, nucleus):
    """Continuum from fused/network-like (low) to punctate (high)."""
    props = _object_props(labels)
    if not props:
        return 0.0
    n = len(props)
    eccentricities = np.array([p.eccentricity for p in props])
    areas = np.array([p.area for p in props], dtype=np.float64)
    # punctate: many small round objects. fused: few elongated objects.
    roundness = 1.0 - eccentricities.mean()
    count_term = np.log1p(n)
    size_cv = areas.std() / areas.mean() if areas.mean() > 0 else 0.0
    return float(roundness * count_term / (1.0 + size_cv))


@_safe
def mitochondrial_neighbor_intensity_variogram_slope(mito, labels, cell, nucleus):
    """Slope of the empirical semi-variogram of mito intensity vs lag."""
    m = (labels > 0) & cell
    if m.sum() < 64:
        m = labels > 0
    if m.sum() < 64:
        return 0.0
    ys, xs = np.nonzero(m)
    vals = mito[m]
    # subsample for speed
    rng = np.random.default_rng(0)
    if ys.size > 2000:
        idx = rng.choice(ys.size, 2000, replace=False)
        ys, xs, vals = ys[idx], xs[idx], vals[idx]
    pts = np.column_stack([ys, xs]).astype(np.float64)
    # pairwise lags on a random subset of pairs
    n_pairs = min(8000, ys.size * 4)
    i = rng.integers(0, ys.size, n_pairs)
    j = rng.integers(0, ys.size, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    lag = np.sqrt(((pts[i] - pts[j]) ** 2).sum(axis=1))
    gamma = 0.5 * (vals[i] - vals[j]) ** 2
    # bin
    maxlag = np.percentile(lag, 80)
    bins = np.linspace(0, maxlag, 8)
    xs_b, ys_b = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        sel = (lag >= a) & (lag < b)
        if sel.sum() < 8:
            continue
        xs_b.append(0.5 * (a + b))
        ys_b.append(gamma[sel].mean())
    if len(xs_b) < 3:
        return 0.0
    slope, _, _, _, _ = stats.linregress(xs_b, ys_b)
    return float(slope)


CODE_FEATURES = {
    "mitochondria_background_signal_heterogeneity": mitochondria_background_signal_heterogeneity,
    "mito_perimeter_intensity_gradient_mean": mito_perimeter_intensity_gradient_mean,
    "background_mito_signal_to_noise": background_mito_signal_to_noise,
    "mitochondria_neighbor_intensity_autocorrelation": mitochondria_neighbor_intensity_autocorrelation,
    "mitochondria_texture_anisotropy_glcm": mitochondria_texture_anisotropy_glcm,
    "mitochondria_long_tubule_fraction": mitochondria_long_tubule_fraction,
    "mitochondrial_area_fraction": mitochondrial_area_fraction,
    "mitochondria_local_autocorrelation_length": mitochondria_local_autocorrelation_length,
    "mito_intensity_autocorrelation_length": mito_intensity_autocorrelation_length,
    "mitochondria_to_cell_intensity_dynamic_range_ratio": mitochondria_to_cell_intensity_dynamic_range_ratio,
    "background_noise_robust_snr": background_noise_robust_snr,
    "mito_intensity_geodesic_variation": mito_intensity_geodesic_variation,
    "mitochondria_local_entropy_mean": mitochondria_local_entropy_mean,
    "mitochondrial_radial_anisotropy_index": mitochondrial_radial_anisotropy_index,
    "mitochondria_network_continuity_score": mitochondria_network_continuity_score,
    "mitochondria_length_width_skewness": mitochondria_length_width_skewness,
    "mitochondria_intensity_entropy": mitochondria_intensity_entropy,
    "mito_clustered_vs_isolated_object_fraction": mito_clustered_vs_isolated_object_fraction,
    "mitochondria_intensity_gini_inside_cell": mitochondria_intensity_gini_inside_cell,
    "mitochondria_area_fraction_in_perinuclear_band": mitochondria_area_fraction_in_perinuclear_band,
    "mitochondria_intensity_entropy_within_cell": mitochondria_intensity_entropy_within_cell,
    "mitochondrial_clusteredness_by_nearest_neighbor": mitochondrial_clusteredness_by_nearest_neighbor,
    "mitochondrial_network_fragmentation_score": mitochondrial_network_fragmentation_score,
    "mitochondrial_neighbor_intensity_variogram_slope": mitochondrial_neighbor_intensity_variogram_slope,
}

VLM_FEATURES = ["mitochondrial_network_compactness"]
