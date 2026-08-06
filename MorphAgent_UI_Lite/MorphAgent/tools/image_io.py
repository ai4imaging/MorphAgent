"""Format-tolerant image / mask loading helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


# Preview / overlay images written next to real masks (Allen, Cellpose, etc.).
_NON_MASK_NAME_TOKENS = (
    "visualization",
    "visualisation",
    "overlay",
    "preview",
    "rgb",
    "color",
    "colour",
    "summary",
)


def is_segmentation_mask_filename(name: str) -> bool:
    """Return False for known non-mask previews that live under segmentation/."""
    stem = Path(name).stem.lower()
    return not any(token in stem for token in _NON_MASK_NAME_TOKENS)


def load_image_array(path: Path | str) -> Any:
    """Load an image/mask array from TIFF, MRC, PNG, JPEG, BMP, GIF, etc.

    Tries tifffile for ``.tif/.tiff`` first (with PIL fallback), then PIL / imageio
    for other formats. Raises ``ValueError`` if nothing can load the file.
    """
    import numpy as np

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")

    suffix = path.suffix.lower()
    errors: list[str] = []

    if suffix in {".tif", ".tiff"}:
        try:
            import tifffile

            return np.asarray(tifffile.imread(str(path)))
        except Exception as exc:  # noqa: BLE001 — try broader loaders next
            errors.append(f"tifffile: {exc}")

    if suffix in {".mrc", ".map", ".rec"}:
        try:
            import mrcfile

            with mrcfile.open(str(path), permissive=True) as handle:
                return np.asarray(handle.data).copy()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"mrcfile: {exc}")

    try:
        from PIL import Image

        arr = np.asarray(Image.open(str(path)))
        if arr.ndim == 3 and arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr
    except Exception as exc:  # noqa: BLE001
        errors.append(f"PIL: {exc}")

    try:
        import imageio.v2 as imageio

        return np.asarray(imageio.imread(str(path)))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"imageio: {exc}")

    detail = "; ".join(errors) if errors else "no loaders available"
    raise ValueError(f"Failed to load image {path.name}: {detail}")


def load_image_array_optional(path: Path | str) -> Optional[Any]:
    """Like ``load_image_array`` but returns None on failure."""
    try:
        return load_image_array(path)
    except Exception:
        return None
