#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert a timelapse / multi-dimensional TIFF into a single 2D intensity image (max projection over leading dimensions such as T, Z, C),
for use with load_mip + Allen segmentation.
"""

from pathlib import Path

import numpy as np
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter


def _squeeze_leading_ones(arr):
    arr = np.asarray(arr, dtype=np.float32)
    while arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _collapse_extra_dims_to_2d(arr):
    """Collapse (..., H, W) down to (H, W): take max over all axes except the last two."""
    arr = _squeeze_leading_ones(arr)
    while arr.ndim > 3:
        arr = np.max(arr, axis=0)
    if arr.ndim == 3:
        d0, d1, d2 = arr.shape
        if d0 <= 8 and d1 >= 32 and d2 >= 32:
            arr = arr[0]
        elif d2 <= 8 and d0 >= 32 and d1 >= 32:
            arr = arr[..., 0]
        else:
            arr = np.max(arr, axis=0)
    if arr.ndim != 2:
        raise ValueError("Could not obtain a 2D image, array shape: %s" % (arr.shape,))
    return arr.astype(np.float32, copy=False)


def read_tif_as_single_2d_mip(path):
    """Treat the whole file as one sample: take max over T/Z/C/S etc. to obtain a single 2D image."""
    img = AICSImage(str(path))
    data = img.get_image_data()
    return _collapse_extra_dims_to_2d(data)


def read_tif_per_time_index(path):
    """
    Split into multiple 2D images along the time dimension (falls back to a single whole-file MIP if the T dimension cannot be parsed).
    Returns [(suffix, ndarray(H,W)), ...], where suffix is e.g. t000 or an empty string.
    """
    img = AICSImage(str(path))
    try:
        data = img.get_image_data("TCZYX")
        t = int(data.shape[0])
        if t > 1:
            out = []
            for ti in range(t):
                plane = _collapse_extra_dims_to_2d(data[ti])
                out.append(("t%04d" % ti, plane))
            return out
    except Exception:
        pass
    try:
        data = np.asarray(img.get_image_data(), dtype=np.float32)
        if data.ndim >= 4 and data.shape[0] > 1:
            h, w = data.shape[-2], data.shape[-1]
            if h >= 32 and w >= 32:
                out = []
                for ti in range(data.shape[0]):
                    plane = _collapse_extra_dims_to_2d(data[ti])
                    out.append(("t%04d" % ti, plane))
                return out
    except Exception:
        pass
    return [("", read_tif_as_single_2d_mip(path))]


def save_2d_float_tiff(path, arr_hw):
    """Write a float32 2D TIFF (single plane, Z=1)."""
    path = Path(path)
    if path.exists():
        path.unlink()
    a = np.asarray(arr_hw, dtype=np.float32)
    vol = a[np.newaxis, ...]
    with OmeTiffWriter(str(path)) as w:
        w.save(vol)
