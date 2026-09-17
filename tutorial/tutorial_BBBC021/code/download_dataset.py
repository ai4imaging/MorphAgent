#!/usr/bin/env python3
"""Download the BBBC021 image dataset into data/dataset/."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List, Optional
from urllib.request import urlopen, urlretrieve

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from paths import DATA_DIR, DATASET_DIR, ROOT

ZENODO_RECORD_ID = "22763120"
ZENODO_RECORD_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}"
EXPECTED_ZIP_SHA256 = "defacc6da001fbc7b4fd6ad725d7cb1e1bc67545983090088691daa3f3c0dcb5"
EXPECTED_ZIP_SIZE = 2_888_492_157


def dataset_ready(dataset_dir: Path = DATASET_DIR) -> bool:
    if not dataset_dir.is_dir():
        return False
    n = sum(1 for p in dataset_dir.iterdir() if p.is_dir() and (p / "image.tif").is_file())
    return n >= 10


def inspect_dataset(dataset_dir: Path = DATASET_DIR) -> dict:
    if not dataset_dir.exists():
        return {"exists": False, "n_samples": 0, "example": None}
    samples = sorted(
        p for p in dataset_dir.iterdir() if p.is_dir() and (p / "image.tif").is_file()
    )
    example = None
    if samples:
        s = samples[0]
        example = {
            "sample_id": s.name,
            "files": sorted(x.name for x in s.iterdir()),
            "has_slices": (s / "slices").is_dir(),
            "has_segmentation": (s / "segmentation").is_dir(),
        }
    return {
        "exists": True,
        "path": str(dataset_dir.resolve()),
        "n_samples": len(samples),
        "example": example,
    }


def zenodo_record_id(url: str) -> Optional[str]:
    text = (url or "").strip()
    if not text:
        return None
    if text.isdigit():
        return text
    match = re.search(r"zenodo\.org/(?:records|doi)/(?:10\.5281/zenodo\.)?(\d+)", text)
    if match:
        return match.group(1)
    match = re.search(r"10\.5281/zenodo\.(\d+)", text)
    if match:
        return match.group(1)
    return None


def _is_direct_zip_url(url: str) -> bool:
    return url.split("?", 1)[0].lower().endswith(".zip")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _progress(prefix: str):
    last = {"pct": -1}

    def hook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        pct = min(100, int(block_num * block_size * 100 / total_size))
        if pct == last["pct"] or (pct < 100 and pct % 10 != 0):
            return
        last["pct"] = pct
        print(f"  {prefix} {pct}%", flush=True)

    return hook


def _list_zenodo_files(record_id: str) -> List[dict]:
    api = f"https://zenodo.org/api/records/{record_id}"
    with urlopen(api, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return list(payload.get("files") or [])


def _download_url(url: str, dest: Path, label: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {label}\n  {url}\n→ {dest}", flush=True)
    urlretrieve(url, dest, reporthook=_progress(label))


def _download_split_zip(record_id: str, zip_path: Path) -> Path:
    files = _list_zenodo_files(record_id)
    parts = sorted(
        (item for item in files if str(item.get("key", "")).startswith("dataset.zip.part_")),
        key=lambda item: str(item["key"]),
    )
    if not parts:
        raise RuntimeError(
            f"Zenodo record {record_id} has no dataset.zip.part_* files. "
            "Pass a direct .zip URL instead."
        )
    part_dir = DATA_DIR / "zenodo_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    part_paths: List[Path] = []
    for i, item in enumerate(parts, 1):
        name = str(item["key"])
        dest = part_dir / name
        url = item.get("links", {}).get("self") or (
            f"https://zenodo.org/records/{record_id}/files/{name}?download=1"
        )
        expected = int(item.get("size") or 0)
        if dest.is_file() and (expected == 0 or dest.stat().st_size == expected):
            print(f"  [{i}/{len(parts)}] {name} already present")
        else:
            _download_url(url, dest, f"[{i}/{len(parts)}] {name}")
            if expected and dest.stat().st_size != expected:
                raise RuntimeError(
                    f"{name} size mismatch: got {dest.stat().st_size}, expected {expected}"
                )
        part_paths.append(dest)

    print(f"Concatenating {len(part_paths)} parts → {zip_path}")
    with zip_path.open("wb") as out:
        for path in part_paths:
            with path.open("rb") as src:
                shutil.copyfileobj(src, out)
    shutil.rmtree(part_dir, ignore_errors=True)
    return zip_path


def _extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    print(f"Extracting {zip_path} ...")
    tmp = DATA_DIR / "_extract_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    candidates = [tmp / "dataset", tmp]
    nested = list(tmp.glob("**/image.tif"))
    if nested:
        candidates.insert(0, nested[0].parent.parent)
    chosen = None
    for cand in candidates:
        n = (
            sum(1 for p in cand.iterdir() if p.is_dir() and (p / "image.tif").is_file())
            if cand.is_dir()
            else 0
        )
        if n >= 10:
            chosen = cand
            break
    if chosen is None:
        raise RuntimeError(f"Could not find sample directories with image.tif inside {zip_path}")
    if dest_dir.exists() or dest_dir.is_symlink():
        if dest_dir.is_symlink() or dest_dir.is_file():
            dest_dir.unlink()
        else:
            shutil.rmtree(dest_dir)
    shutil.move(str(chosen), str(dest_dir))
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    print("Done:", inspect_dataset(dest_dir))
    return dest_dir


def download_and_extract(url: str = ZENODO_RECORD_URL, dest_dir: Path = DATASET_DIR) -> Path:
    url = (url or "").strip() or ZENODO_RECORD_URL
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "bbbc021_dataset.zip"

    if zip_path.is_file() and zip_path.stat().st_size == EXPECTED_ZIP_SIZE:
        print(f"Reusing existing {zip_path}")
    elif _is_direct_zip_url(url):
        _download_url(url, zip_path, "dataset.zip")
    else:
        record_id = zenodo_record_id(url)
        if not record_id:
            raise ValueError(
                "Pass a Zenodo record URL (https://zenodo.org/records/22763120) "
                "or a direct .zip file URL."
            )
        _download_split_zip(record_id, zip_path)

    using_known_archive = (
        zip_path.stat().st_size == EXPECTED_ZIP_SIZE
        or zenodo_record_id(url) == ZENODO_RECORD_ID
        or not _is_direct_zip_url(url)
    )
    if using_known_archive:
        if zip_path.stat().st_size != EXPECTED_ZIP_SIZE:
            print(
                f"Warning: zip size is {zip_path.stat().st_size} bytes "
                f"(README lists {EXPECTED_ZIP_SIZE}). Checking SHA256 anyway."
            )
        digest = _sha256_file(zip_path)
        if digest != EXPECTED_ZIP_SHA256:
            raise RuntimeError(
                f"SHA256 mismatch for {zip_path}:\n  got      {digest}\n"
                f"  expected {EXPECTED_ZIP_SHA256}"
            )
        print("SHA256 OK")
    return _extract_zip(zip_path, dest_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default=ZENODO_RECORD_URL,
        help="Zenodo record page or a direct .zip URL",
    )
    args = parser.parse_args()
    print("tutorial root:", ROOT)
    download_and_extract(args.url)


if __name__ == "__main__":
    main()
