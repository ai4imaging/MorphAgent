"""Sandbox-side runner for LLM-authored fallback segmentation.

Invoked as a standalone script by :mod:`tools.auto_segmentation` in the same
interpreter that executes generated feature code. It imports nothing from the
MorphAgent package so it stays runnable inside the isolated sandbox env; only
numpy plus an image writer (tifffile / imageio / PIL) is required.

Usage::

    python auto_segmentation_runner.py --code segment_code.py --image image.tif \
        --out-dir sample/segmentation --names-json names.json \
        [--preview-dir previews] [--strict]

A single JSON object is printed to stdout; stderr carries diagnostics only.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

# Distinct overlay colours, ordered so the first few stay readable on grey.
_COLORS = (
    (255, 72, 72),
    (72, 220, 96),
    (96, 160, 255),
    (255, 208, 64),
    (232, 96, 255),
)

# A mask that is empty or covers essentially everything carries no information.
_MIN_FOREGROUND_FRACTION = 1e-5
_MAX_FOREGROUND_FRACTION = 0.995


def load_image(path: Path) -> np.ndarray:
    """Load TIFF / MRC / PNG / JPEG / BMP into an array, trying several backends."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"image not found: {path}")

    suffix = path.suffix.lower()
    errors: list[str] = []

    if suffix in {".tif", ".tiff"}:
        try:
            import tifffile

            return np.asarray(tifffile.imread(str(path)))
        except Exception as exc:  # noqa: BLE001 - fall through to other loaders
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

    raise ValueError(f"failed to load {path.name}: {'; '.join(errors)}")


def to_display_plane(arr: np.ndarray) -> np.ndarray:
    """Reduce any image layout to the single 2-D plane masks are defined on.

    Small leading/trailing axes are treated as channels and averaged; a long
    axis is treated as a z-stack and max-projected. The result defines the
    mask shape contract shared by the prompt, the validator and the overlay.
    """
    plane = np.squeeze(np.asarray(arr))
    if plane.ndim <= 2:
        return np.atleast_2d(plane)
    if plane.ndim == 3:
        axis = int(np.argmin(plane.shape))
        if plane.shape[axis] <= 5:
            return plane.astype(np.float64).mean(axis=axis)
        return plane.max(axis=0)
    while plane.ndim > 2:
        plane = plane.max(axis=0)
    return plane


def stretch_to_uint8(plane: np.ndarray) -> np.ndarray:
    """Percentile contrast stretch so faint fluorescence is visible to the VLM."""
    values = np.asarray(plane, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    low, high = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(values.min()), float(values.max())
    if high <= low:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (values - low) / (high - low)
    return (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


def _shift(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Shift a 2-D array, filling vacated pixels with zeros (no wrap-around)."""
    out = np.zeros_like(arr)
    height, width = arr.shape
    src_y = slice(max(0, -dy), height - max(0, dy))
    dst_y = slice(max(0, dy), height - max(0, -dy))
    src_x = slice(max(0, -dx), width - max(0, dx))
    dst_x = slice(max(0, dx), width - max(0, -dx))
    out[dst_y, dst_x] = arr[src_y, src_x]
    return out


def label_boundary(mask: np.ndarray) -> np.ndarray:
    """Outline pixels of a binary or instance-label mask, via 4-neighbour differences."""
    labels = np.asarray(mask).astype(np.int64, copy=False)
    edge = np.zeros(labels.shape, dtype=bool)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        edge |= _shift(labels, dy, dx) != labels
    return edge & (labels > 0)


def normalise_mask(value: object, spatial_shape: tuple[int, int]) -> np.ndarray:
    """Coerce whatever ``segment`` returned into a 2-D integer mask of the right shape."""
    arr = np.squeeze(np.asarray(value))
    if arr.ndim != 2:
        raise ValueError(
            f"mask must be 2-D after squeezing, got shape {tuple(np.shape(arr))}"
        )
    if tuple(arr.shape) != tuple(spatial_shape):
        raise ValueError(
            f"mask shape {tuple(arr.shape)} does not match the image plane "
            f"{tuple(spatial_shape)}"
        )

    if arr.dtype == bool:
        return arr.astype(np.uint8)

    if np.issubdtype(arr.dtype, np.floating):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        rounded = np.rint(arr)
        if np.allclose(arr, rounded, atol=1e-6):
            arr = rounded
        else:
            # A probability / distance map: threshold it rather than guess labels.
            return (arr > 0.5).astype(np.uint8)

    arr = np.asarray(arr, dtype=np.int64)
    arr = np.where(arr < 0, 0, arr)
    top = int(arr.max()) if arr.size else 0
    if top <= 1:
        return arr.astype(np.uint8)
    if top <= 65535:
        return arr.astype(np.uint16)
    return arr.astype(np.uint32)


def mask_statistics(mask: np.ndarray) -> dict:
    """Summarise a mask for the LLM feedback loop and the run artifacts."""
    foreground = mask > 0
    total = int(mask.size) or 1
    labels = np.unique(mask)
    return {
        "foreground_fraction": round(float(foreground.sum()) / total, 6),
        "object_count": int(max(0, labels.size - 1)) if labels.size else 0,
        "max_label": int(mask.max()) if mask.size else 0,
        "dtype": str(mask.dtype),
        "shape": list(mask.shape),
    }


def degeneracy_problem(name: str, stats: dict) -> str | None:
    """Return a human-readable complaint when a mask carries no usable signal."""
    fraction = stats["foreground_fraction"]
    if fraction <= _MIN_FOREGROUND_FRACTION:
        return (
            f"'{name}' is empty (foreground fraction {fraction:.6f}); the threshold "
            "is far too strict or the wrong channel was used"
        )
    if fraction >= _MAX_FOREGROUND_FRACTION:
        return (
            f"'{name}' covers {fraction * 100:.1f}% of the frame, so it does not "
            "delineate objects; the threshold is far too permissive"
        )
    return None


def _name_core(name: str) -> str:
    """Strip the ``mask_``/``seg_`` decoration so 'mask_nucleus' can meet 'nuclei'."""
    core = str(name).strip().lower()
    for prefix in ("mask_", "mask", "seg_", "segmentation_"):
        if core.startswith(prefix) and len(core) > len(prefix):
            core = core[len(prefix) :]
            break
    return core.strip("_")


def resolve_keys(returned: dict, expected: list[str]) -> tuple[dict, list[str]]:
    """Map the dict ``segment`` returned onto the planned mask names.

    Exact keys win, then case-insensitive, then a fuzzy match on the meaningful
    part of the name ('nuclei' -> 'mask_nucleus'). Positional order is only used
    when exactly one planned mask and one returned key are left, since matching
    several by position silently swaps their meaning. Every non-exact match is
    reported so the decision stays auditable.
    """
    import difflib

    notes: list[str] = []
    resolved: dict[str, object] = {}
    remaining = dict(returned)

    for name in expected:
        if name in remaining:
            resolved[name] = remaining.pop(name)

    lowered = {str(key).lower(): key for key in remaining}
    for name in expected:
        if name in resolved:
            continue
        actual = lowered.get(name.lower())
        if actual is not None and actual in remaining:
            resolved[name] = remaining.pop(actual)
            notes.append(f"matched returned key '{actual}' to planned mask '{name}' (case-insensitive)")

    for name in expected:
        if name in resolved or not remaining:
            continue
        cores = {_name_core(key): key for key in remaining}
        close = difflib.get_close_matches(_name_core(name), list(cores), n=1, cutoff=0.7)
        if close:
            actual = cores[close[0]]
            resolved[name] = remaining.pop(actual)
            notes.append(f"matched returned key '{actual}' to planned mask '{name}' by name similarity")

    missing = [name for name in expected if name not in resolved]
    if len(missing) == 1 and len(remaining) == 1:
        actual, value = next(iter(remaining.items()))
        resolved[missing[0]] = value
        remaining.pop(actual)
        notes.append(
            f"matched the only remaining returned key '{actual}' to the only "
            f"remaining planned mask '{missing[0]}'"
        )

    for leftover in remaining:
        notes.append(f"ignored unexpected key '{leftover}'")

    return resolved, notes


def write_mask(path: Path, mask: np.ndarray) -> None:
    """Persist a mask as TIFF, falling back through the available writers."""
    errors: list[str] = []
    try:
        import tifffile

        tifffile.imwrite(str(path), mask)
        return
    except Exception as exc:  # noqa: BLE001
        errors.append(f"tifffile: {exc}")
    try:
        import imageio.v2 as imageio

        imageio.imwrite(str(path), mask)
        return
    except Exception as exc:  # noqa: BLE001
        errors.append(f"imageio: {exc}")
    try:
        from PIL import Image

        Image.fromarray(mask).save(str(path))
        return
    except Exception as exc:  # noqa: BLE001
        errors.append(f"PIL: {exc}")
    raise ValueError(f"failed to write {path.name}: {'; '.join(errors)}")


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Persist an RGB preview for the VLM check."""
    try:
        from PIL import Image

        Image.fromarray(rgb).save(str(path))
        return
    except Exception:  # noqa: BLE001
        pass
    import imageio.v2 as imageio

    imageio.imwrite(str(path), rgb)


def render_overlay(base: np.ndarray, masks: dict, names: list[str]) -> np.ndarray:
    """Tint mask interiors and draw solid outlines over the stretched raw image."""
    rgb = np.dstack([base] * 3).astype(np.float64)
    for index, name in enumerate(names):
        mask = masks.get(name)
        if mask is None:
            continue
        color = np.asarray(_COLORS[index % len(_COLORS)], dtype=np.float64)
        interior = mask > 0
        if interior.any():
            rgb[interior] = rgb[interior] * 0.7 + color * 0.3
        edge = label_boundary(mask)
        if edge.any():
            rgb[edge] = color
    return np.clip(rgb, 0.0, 255.0).astype(np.uint8)


def load_segment_function(code_path: Path):
    """Execute the generated module and hand back its ``segment`` callable."""
    source = Path(code_path).read_text(encoding="utf-8")
    namespace: dict = {"__name__": "morphagent_generated_segmentation"}
    exec(compile(source, str(code_path), "exec"), namespace)  # noqa: S102
    func = namespace.get("segment")
    if not callable(func):
        raise ValueError("the generated code does not define a callable segment(img)")
    return func


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", required=True, help="Python file defining segment(img)")
    parser.add_argument("--image", required=True, help="Image to segment")
    parser.add_argument("--out-dir", required=True, help="Directory to write masks into")
    parser.add_argument("--names-json", required=True, help="JSON file with planned mask names")
    parser.add_argument("--preview-dir", default="", help="Where to write PNG previews")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when a mask is missing or degenerate (used for the trial sample)",
    )
    args = parser.parse_args()

    expected = json.loads(Path(args.names_json).read_text(encoding="utf-8"))
    if not isinstance(expected, list) or not expected:
        raise ValueError("--names-json must contain a non-empty list of mask names")
    expected = [str(name) for name in expected]

    image = load_image(Path(args.image))
    plane = to_display_plane(image)
    spatial_shape = tuple(int(size) for size in plane.shape)

    segment = load_segment_function(Path(args.code))
    returned = segment(image)

    if not isinstance(returned, dict):
        if len(expected) == 1:
            returned = {expected[0]: returned}
        else:
            raise ValueError(
                f"segment(img) must return a dict keyed by {expected}, got "
                f"{type(returned).__name__}"
            )
    if not returned:
        raise ValueError("segment(img) returned an empty dict")

    resolved, notes = resolve_keys(returned, expected)
    missing = [name for name in expected if name not in resolved]
    if missing and args.strict:
        raise ValueError(
            f"segment(img) did not return mask(s) {missing}; returned keys were "
            f"{sorted(str(key) for key in returned)}"
        )
    if not resolved:
        raise ValueError(
            "none of the returned keys could be matched to the planned masks "
            f"{expected}; returned keys were {sorted(str(key) for key in returned)}"
        )

    masks: dict[str, np.ndarray] = {}
    stats: dict[str, dict] = {}
    problems: list[str] = []
    for name in expected:
        if name not in resolved:
            problems.append(f"'{name}' was not produced")
            continue
        mask = normalise_mask(resolved[name], spatial_shape)
        masks[name] = mask
        stats[name] = mask_statistics(mask)
        problem = degeneracy_problem(name, stats[name])
        if problem:
            problems.append(problem)

    if problems and args.strict:
        raise ValueError("; ".join(problems))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for name, mask in masks.items():
        target = out_dir / f"{name}.tif"
        write_mask(target, mask)
        written.append(str(target))

    preview: dict[str, str] = {}
    if args.preview_dir:
        preview_dir = Path(args.preview_dir)
        preview_dir.mkdir(parents=True, exist_ok=True)
        base = stretch_to_uint8(plane)
        original_path = preview_dir / "original.png"
        write_png(original_path, np.dstack([base] * 3))
        preview["original"] = str(original_path)
        ordered = [name for name in expected if name in masks]
        for name in ordered:
            path = preview_dir / f"overlay_{name}.png"
            write_png(path, render_overlay(base, masks, [name]))
            preview[name] = str(path)
        if len(ordered) > 1:
            combined = preview_dir / "overlay_all.png"
            write_png(combined, render_overlay(base, masks, ordered))
            preview["all"] = str(combined)

    print(
        json.dumps(
            {
                "success": True,
                "image": str(Path(args.image)),
                "plane_shape": list(spatial_shape),
                "masks": stats,
                "written": written,
                "preview": preview,
                "problems": problems,
                "notes": notes,
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 - the caller parses this JSON
        print(
            json.dumps(
                {
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
        )
        sys.exit(1)
