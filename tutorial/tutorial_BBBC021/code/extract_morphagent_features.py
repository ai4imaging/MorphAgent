#!/usr/bin/env python3
"""Replay the 467 MorphAgent features from the vendored `source/feature_library`.

Each named feature is either a self-contained `extract.py` (code) or a VLM
planner record. Image data is expected at `data/dataset/` (Zenodo download).
"""
from __future__ import annotations

import argparse
import base64
import inspect
import json
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tifffile

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from paths import (
    DATASET_DIR,
    FEATURE_LIBRARY,
    NAMES_CSV,
    OUTPUT_DIR,
    ROOT,
    VLM_TEMPLATE,
)

LIBRARY_DIR = FEATURE_LIBRARY
OUTPUT_SMOKE = OUTPUT_DIR / "smoke"
FULL_DATASET = DATASET_DIR
TUTORIAL_DIR = ROOT

# Filled at runtime by the notebook / CLI. This tutorial never ships a key.
_VLM_API: Dict[str, Optional[str]] = {
    "base_url": None,
    "api_key": None,
    "model": None,
}

DATASET_CONTEXT = """BBBC021 Cell Painting of MCF-7 breast cancer cells.
Each sample is a 2D RGB uint8 image of shape (512, 512, 3):
  channel 0 / Red   = F-actin (cytoskeleton)
  channel 1 / Green = β-tubulin (microtubules)
  channel 2 / Blue  = DAPI (nuclear DNA)
VLM inputs are the three per-channel PNG slices under slices/.
"""

def load_467_names() -> List[str]:
    df = pd.read_csv(NAMES_CSV)
    col = "feature_name" if "feature_name" in df.columns else df.columns[0]
    names = df[col].astype(str).tolist()
    if len(names) != 467:
        raise ValueError(f"Expected 467 names, found {len(names)}")
    return names


def configure_vlm_api(
    base_url: str,
    api_key: str,
    model: str = "",
) -> None:
    """Register the caller's own OpenAI-compatible vision endpoint.

    Nothing is bundled with this tutorial. `base_url` should look like
    `https://<host>/v1` and accept POST `/chat/completions` with image_url parts.
    """
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


def _require_vlm_api() -> Tuple[str, str, str]:
    if not vlm_api_ready():
        raise RuntimeError(
            "No VLM credentials. In the notebook, paste VLM_API_BASE_URL and "
            "VLM_API_KEY into the credentials cell. This tutorial does not ship a key."
        )
    return str(_VLM_API["base_url"]), str(_VLM_API["api_key"]), str(_VLM_API["model"] or "gpt-4o")


def require_dataset() -> Path:
    if not DATASET_DIR.is_dir():
        raise FileNotFoundError(
            f"Image dataset not found at {DATASET_DIR}. "
            "Run notebook/01_setup_environment_and_data.ipynb first."
        )
    return DATASET_DIR


def list_sample_ids(dataset_root: Path, limit: Optional[int] = None) -> List[str]:
    ids = sorted(
        p.name for p in dataset_root.iterdir()
        if p.is_dir() and (p / "image.tif").is_file()
    )
    if limit is not None:
        ids = ids[: int(limit)]
    return ids


def load_image(sample_dir: Path) -> np.ndarray:
    return np.asarray(tifffile.imread(str(sample_dir / "image.tif")))


def load_segmentation(sample_dir: Path) -> Dict[str, np.ndarray]:
    seg_dir = sample_dir / "segmentation"
    out: Dict[str, np.ndarray] = {}
    if not seg_dir.is_dir():
        return out
    for path in sorted(seg_dir.glob("*.tif")):
        out[path.stem] = np.asarray(tifffile.imread(str(path)))
    return out


def slice_pngs(sample_dir: Path) -> List[str]:
    slices = sample_dir / "slices"
    if not slices.is_dir():
        return []
    return [str(p) for p in sorted(slices.glob("*.png"))]


def compile_extract(code_path: Path):
    text = code_path.read_text(encoding="utf-8")
    namespace: Dict[str, Any] = {"np": np, "__name__": "__feature__"}
    exec(compile(text, str(code_path), "exec"), namespace, namespace)
    func = namespace.get("extract") or namespace.get("extract_all")
    if func is None:
        raise ValueError(f"{code_path} has no extract/extract_all")
    return func


def call_extract(func, image: np.ndarray, seg: Dict[str, np.ndarray]) -> Any:
    params = list(inspect.signature(func).parameters.keys())
    if len(params) == 1:
        return func(image)
    second = params[1]
    if second in {"seg", "segmentation"}:
        return func(image, seg)
    masks = [seg[k] for k in sorted(seg)] if seg else []
    if len(params) == 2:
        return func(image, masks[0] if masks else None)
    args = masks[: max(0, len(params) - 1)]
    while len(args) < len(params) - 1:
        args.append(None)
    return func(image, *args)


def to_scalar(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, dict):
        if len(value) == 1:
            return to_scalar(next(iter(value.values())))
        return None
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        value = value.item()
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def run_code_feature(
    feature_name: str,
    dataset_root: Path,
    sample_ids: Sequence[str],
) -> pd.DataFrame:
    code_path = LIBRARY_DIR / "code" / feature_name / "extract.py"
    if not code_path.is_file():
        raise FileNotFoundError(code_path)
    func = compile_extract(code_path)
    rows = []
    t0 = time.time()
    for i, sid in enumerate(sample_ids, 1):
        sample_dir = dataset_root / sid
        err = None
        val = None
        try:
            image = load_image(sample_dir)
            seg = load_segmentation(sample_dir)
            raw = call_extract(func, image, seg)
            val = to_scalar(raw)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
        rows.append({"sample_id": sid, feature_name: val, "error": err})
        if i == 1 or i == len(sample_ids) or i % 10 == 0:
            print(f"  [{feature_name}] {i}/{len(sample_ids)}")
    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    df.attrs["elapsed_sec"] = elapsed
    print(f"  {feature_name}: {len(sample_ids)} samples in {elapsed:.2f}s "
          f"({elapsed / max(len(sample_ids), 1):.3f}s/sample)")
    return df


def _fill_vlm_prompt(feature: Dict[str, Any], image_paths: Sequence[str]) -> str:
    template = json.loads(VLM_TEMPLATE.read_text(encoding="utf-8"))["template"]
    names = [Path(p).name for p in image_paths]
    image_list = "Images provided (in order):\n" + "\n".join(f"- {n}" for n in names)
    mapping = {
        "dataset_description": DATASET_CONTEXT,
        "dataset_image_format": "Each slice PNG is a single fluorescence channel, 512x512.",
        "channel_information": "Actin (R), Tubulin (G), DAPI (B). PNG filenames encode the channel.",
        "image_list_description": image_list,
        "deep_research_info": "",
        "rag_knowledge_info": "",
        "expert_knowledge_info": "",
        "feature_name": feature.get("name") or feature.get("feature_name"),
        "feature_description": feature.get("description", ""),
        "feature_category": feature.get("category", ""),
        "num_images_provided": str(len(image_paths)),
        "image_info_list": image_list,
    }
    prompt = template
    for key, val in mapping.items():
        prompt = prompt.replace("{" + key + "}", str(val))
    prompt = re.sub(r"\{[a-zA-Z0-9_]+\}", "", prompt)
    return prompt


def parse_vlm_score(text: str) -> Optional[float]:
    patterns = [
        r'\{\s*"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*\}',
        r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
    ]
    for pat in patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        if matches:
            score = float(matches[-1])
            if 0.0 <= score <= 100.0:
                return score
    return None


def call_vlm_api(prompt: str, image_paths: Sequence[str], timeout: int = 150) -> Tuple[Optional[float], str]:
    base_url, api_key, model = _require_vlm_api()
    content: List[Dict[str, Any]] = []
    for path in image_paths:
        raw = Path(path).read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        suffix = Path(path).suffix.lower().lstrip(".")
        mime = "jpeg" if suffix in {"jpg", "jpeg"} else "png"
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/{mime};base64,{b64}"},
        })
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1024,
        "temperature": 0.0,
    }
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "MorphAgent/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"VLM HTTP {exc.code}: {detail[:500]}") from exc
    text = body["choices"][0]["message"]["content"] or ""
    return parse_vlm_score(text), text


def run_vlm_feature(
    feature_name: str,
    dataset_root: Path,
    sample_ids: Sequence[str],
) -> pd.DataFrame:
    feat_path = LIBRARY_DIR / "vlm" / feature_name / "feature.json"
    if not feat_path.is_file():
        raise FileNotFoundError(feat_path)
    feature = json.loads(feat_path.read_text(encoding="utf-8"))
    rows = []
    t0 = time.time()
    for i, sid in enumerate(sample_ids, 1):
        pngs = slice_pngs(dataset_root / sid)
        err = None
        val = None
        response = None
        try:
            if not pngs:
                raise FileNotFoundError(f"no PNG slices for {sid}")
            prompt = _fill_vlm_prompt(feature, pngs)
            val, response = call_vlm_api(prompt, pngs)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            response = traceback.format_exc()[-2000:]
        rows.append({
            "sample_id": sid,
            feature_name: val,
            "error": err,
            "response_tail": (response or "")[-600:],
        })
        print(f"  [{feature_name}] {i}/{len(sample_ids)} -> {val} {err or ''}")
    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    df.attrs["elapsed_sec"] = elapsed
    print(f"  {feature_name}: {len(sample_ids)} samples in {elapsed:.2f}s "
          f"({elapsed / max(len(sample_ids), 1):.1f}s/sample)")
    return df


def estimate_full_runtime(
    code_sec_per_sample: float,
    vlm_sec_per_sample: float,
    n_code: int = 438,
    n_vlm: int = 29,
    n_images: int = 3552,
) -> Dict[str, float]:
    code_hours = n_code * n_images * code_sec_per_sample / 3600.0
    vlm_hours = n_vlm * n_images * vlm_sec_per_sample / 3600.0
    return {
        "n_images": n_images,
        "n_code": n_code,
        "n_vlm": n_vlm,
        "code_sec_per_sample": code_sec_per_sample,
        "vlm_sec_per_sample": vlm_sec_per_sample,
        "code_hours_serial": code_hours,
        "vlm_hours_serial": vlm_hours,
        "total_hours_serial": code_hours + vlm_hours,
        "code_hours_16way": code_hours / 16.0,
        "vlm_hours_if_batched_per_image": n_images * vlm_sec_per_sample / 3600.0,
    }


def smoke_test(
    n_code_samples: int = 5,
    n_vlm_samples: int = 2,
    code_feature: str = "tubulin_intensity_total",
    vlm_feature: str = "vlm_nuclear_cap_presence",
) -> Dict[str, Any]:
    dataset = require_dataset()
    if not (LIBRARY_DIR / "manifest.csv").is_file():
        raise FileNotFoundError(f"Missing {LIBRARY_DIR / 'manifest.csv'}")
    OUTPUT_SMOKE.mkdir(parents=True, exist_ok=True)
    sample_ids = list_sample_ids(dataset)
    print(f"Dataset: {dataset} ({len(sample_ids)} samples)")

    code_ids = sample_ids[:n_code_samples]
    print(f"\n=== CODE smoke: {code_feature} on {len(code_ids)} samples ===")
    code_df = run_code_feature(code_feature, dataset, code_ids)
    code_df.to_csv(OUTPUT_SMOKE / f"smoke_code_{code_feature}.csv", index=False)

    print(f"\n=== VLM smoke: {vlm_feature} on {n_vlm_samples} samples ===")
    vlm_df = None
    vlm_dt = float("nan")
    if not vlm_api_ready():
        print("Skipping VLM smoke: fill in your own base URL and API key first.")
    else:
        vlm_ids = sample_ids[:n_vlm_samples]
        vlm_df = run_vlm_feature(vlm_feature, dataset, vlm_ids)
        vlm_df.to_csv(OUTPUT_SMOKE / f"smoke_vlm_{vlm_feature}.csv", index=False)
        vlm_dt = float(vlm_df.attrs.get("elapsed_sec", 0.0)) / max(len(vlm_ids), 1)

    code_dt = float(code_df.attrs.get("elapsed_sec", 0.0)) / max(len(code_ids), 1)
    if not np.isfinite(vlm_dt):
        print("Runtime estimate needs a successful VLM smoke; code-only timing is "
              f"{code_dt:.3f}s/sample.")
        estimate = estimate_full_runtime(code_dt, 14.0)
        estimate["vlm_sec_per_sample_note"] = "placeholder 14s; VLM was skipped"
    else:
        estimate = estimate_full_runtime(code_dt, vlm_dt)
    (OUTPUT_SMOKE / "full_runtime_estimate.json").write_text(
        json.dumps(estimate, indent=2), encoding="utf-8"
    )
    print("\nFull-dataset serial estimate (hours):",
          {k: round(v, 2) for k, v in estimate.items() if "hours" in k})
    return {"code": code_df, "vlm": vlm_df, "estimate": estimate}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--code-feature", default="tubulin_intensity_total")
    parser.add_argument("--vlm-feature", default="vlm_nuclear_cap_presence")
    parser.add_argument("--n-code-samples", type=int, default=5)
    parser.add_argument("--n-vlm-samples", type=int, default=2)
    parser.add_argument(
        "--api-base",
        default=os.environ.get("MORPHAGENT_VLM_API_BASE", ""),
        help="OpenAI-compatible VLM base URL (or env MORPHAGENT_VLM_API_BASE)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MORPHAGENT_VLM_API_KEY", ""),
        help="VLM API key (or env MORPHAGENT_VLM_API_KEY; never shipped)",
    )
    parser.add_argument(
        "--api-model",
        default=os.environ.get("MORPHAGENT_VLM_MODEL", ""),
        help="Vision model name (or env MORPHAGENT_VLM_MODEL)",
    )
    args = parser.parse_args()
    if args.api_base or args.api_key:
        configure_vlm_api(args.api_base, args.api_key, args.api_model)
    if args.smoke:
        smoke_test(
            n_code_samples=args.n_code_samples,
            n_vlm_samples=args.n_vlm_samples,
            code_feature=args.code_feature,
            vlm_feature=args.vlm_feature,
        )


if __name__ == "__main__":
    main()
