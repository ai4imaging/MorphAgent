"""Replay the locked 25 MorphAgent HSC features.

24 code extractors live in source/feature_library/code/.
1 VLM feature (mitochondrial_network_compactness) needs the reviewer's own
OpenAI-compatible vision endpoint.

A full pass over discovery (110) + validation (162) images is documented
but not launched here. The notebooks run a synthetic smoke test.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from skimage.draw import disk as draw_disk

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from paths import (
    CATALOG_CSV,
    DATASET_DIR,
    FEATURE_LIBRARY,
    NAMES_CSV,
    OUTPUT_DIR,
)

OUTPUT_SMOKE = OUTPUT_DIR / "smoke"

_VLM_API: Dict[str, Optional[str]] = {
    "base_url": None,
    "api_key": None,
    "model": None,
}


def load_25_names() -> List[str]:
    df = pd.read_csv(NAMES_CSV)
    col = "feature_name" if "feature_name" in df.columns else df.columns[0]
    names = df[col].astype(str).tolist()
    if len(names) != 25:
        raise ValueError(f"Expected 25 names, found {len(names)}")
    return names


def load_catalog() -> pd.DataFrame:
    return pd.read_csv(CATALOG_CSV)


def configure_vlm_api(base_url: str, api_key: str, model: str = "") -> None:
    base = (base_url or "").strip().rstrip("/")
    key = (api_key or "").strip()
    mdl = (model or "").strip()
    if not base or not key:
        raise ValueError("Both VLM base URL and API key are required")
    if not base.endswith("/v1"):
        base = base + "/v1"
    _VLM_API["base_url"] = base
    _VLM_API["api_key"] = key
    _VLM_API["model"] = mdl or "gpt-4o"
    print(f"VLM endpoint configured: {_VLM_API['base_url']}  model={_VLM_API['model']}")


def vlm_api_ready() -> bool:
    return bool(_VLM_API.get("base_url") and _VLM_API.get("api_key"))


def _load_extract(path: Path):
    spec = importlib.util.spec_from_file_location(path.parent.name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "extract"):
        raise AttributeError(f"{path} has no extract()")
    return mod.extract


def extract_code_feature(name: str, img, *masks) -> float:
    path = FEATURE_LIBRARY / "code" / name / "extract.py"
    fn = _load_extract(path)
    return float(fn(img, *masks))


def make_synthetic_cell(kind: str = "network", size: int = 192, seed: int = 0):
    """Build a toy HSC-like cell for extractor smoke tests (not paper data)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float64)
    yy, xx = np.ogrid[:size, :size]
    cy, cx = size / 2, size / 2
    cell = (yy - cy) ** 2 + (xx - cx) ** 2 <= (size * 0.42) ** 2
    nucleus = (yy - cy) ** 2 + (xx - cx) ** 2 <= (size * 0.14) ** 2
    labels = np.zeros((size, size), dtype=np.int32)
    lid = 1
    if kind == "network":
        # a few elongated tubules
        for k, (angle, length) in enumerate([(0.3, 50), (1.2, 40), (2.1, 45), (2.8, 35)]):
            for t in np.linspace(-length, length, int(length * 2)):
                y = int(cy + 8 * np.sin(k) + t * np.sin(angle))
                x = int(cx + 8 * np.cos(k) + t * np.cos(angle))
                rr, cc = draw_disk((y, x), 2, shape=img.shape)
                img[rr, cc] = 0.7 + 0.2 * rng.random()
                labels[rr, cc] = lid
            lid += 1
    else:
        # punctate fragments
        for _ in range(18):
            y = int(rng.integers(int(size * 0.25), int(size * 0.75)))
            x = int(rng.integers(int(size * 0.25), int(size * 0.75)))
            rr, cc = draw_disk((y, x), int(rng.integers(2, 5)), shape=img.shape)
            img[rr, cc] = 0.5 + 0.4 * rng.random()
            labels[rr, cc] = lid
            lid += 1
    img[~cell] = 0
    labels[~cell] = 0
    img += 0.03 * rng.standard_normal(img.shape)
    img = np.clip(img, 0, 1)
    return img, labels, cell.astype(np.uint8), nucleus.astype(np.uint8)


def smoke_code_features(n_synthetic: int = 2) -> pd.DataFrame:
    catalog = load_catalog()
    code_names = catalog.loc[catalog["method"] == "code", "feature_name"].tolist()
    rows = []
    for i, kind in enumerate(["network", "punctate"][:n_synthetic]):
        img, labels, cell, nucleus = make_synthetic_cell(kind=kind, seed=i)
        rec: Dict[str, Any] = {"sample_id": f"synthetic_{kind}", "kind": kind}
        for name in code_names:
            rec[name] = extract_code_feature(name, img, labels, cell, nucleus)
        rows.append(rec)
    df = pd.DataFrame(rows)
    OUTPUT_SMOKE.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_SMOKE / "synthetic_code_features.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}  shape={df.shape}")
    return df


def list_real_samples(limit: Optional[int] = None) -> List[str]:
    if not DATASET_DIR.is_dir():
        return []
    ids = sorted(
        p.name
        for p in DATASET_DIR.iterdir()
        if p.is_dir() and (p / "image.tif").is_file()
    )
    if limit is not None:
        ids = ids[: int(limit)]
    return ids


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="run synthetic code-feature smoke")
    args = p.parse_args(argv)
    if args.smoke:
        smoke_code_features()
        return 0
    print("Pass --smoke to run the synthetic extractor check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
