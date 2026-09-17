#!/usr/bin/env python3
"""Write the three public notebooks under notebook/."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = ROOT / "notebook"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

KS = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "pygments_lexer": "ipython3"},
}

BOOTSTRAP = r"""
from pathlib import Path
import sys

HERE = Path.cwd().resolve()
if HERE.name == "notebook":
    TUTORIAL = HERE.parent
elif (HERE / "code").is_dir() and (HERE / "source").is_dir():
    TUTORIAL = HERE
else:
    TUTORIAL = HERE
sys.path.insert(0, str(TUTORIAL / "code"))

import importlib
import paths as P
P = importlib.reload(P)
print("Tutorial root:", P.ROOT)
print("data/        :", P.DATA_DIR)
print("source/      :", P.SOURCE_DIR)
print("code/        :", P.CODE_DIR)
"""


def new_nb():
    nb = nbf.v4.new_notebook()
    nb.metadata = dict(KS)
    return nb, []


def md(cells, text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n") + "\n"))


def code(cells, text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n") + "\n"))


def write(nb, cells, name):
    nb.cells = cells
    out = NOTEBOOK_DIR / name
    out.write_text(nbf.writes(nb), encoding="utf-8")
    print("Wrote", out, "n_cells=", len(cells))


# ---------------------------------------------------------------------------
# 01 setup
# ---------------------------------------------------------------------------
nb, cells = new_nb()
md(cells, r"""
# 01 — Environment and BBBC021 image data

This folder is self-contained except for the **images**. The 467 MorphAgent feature implementations live in `source/feature_library/` (no other MorphAgent checkout is required).

1. Install Python dependencies.
2. Download the BBBC021 image archive from Zenodo ([record 22763120](https://zenodo.org/records/22763120), DOI [10.5281/zenodo.22763120](https://doi.org/10.5281/zenodo.22763120)) into `data/dataset/`.
3. Check that each sample has `image.tif`, `slices/` and `segmentation/`.

If `data/dataset/` already exists, the download cell is skipped. The Zenodo upload is split into `dataset.zip.part_*`; the download helper concatenates those parts before extracting.
""")
md(cells, r"""
## 0. Locate the tutorial root

Run this notebook from `notebook/` or from the tutorial root.
""")
code(cells, BOOTSTRAP)
md(cells, r"""
## 1. Install dependencies
""")
code(cells, r"""
import subprocess, sys
req = P.ROOT / "requirements.txt"
print("Installing", req)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])
""")
md(cells, r"""
## 2. Download images from Zenodo

The published archive lives at [zenodo.org/records/22763120](https://zenodo.org/records/22763120). Leave `ZENODO_RECORD_URL` as-is unless you are pointing at another record. If `data/dataset/` already exists, this cell is skipped.
""")
code(cells, r"""
ZENODO_RECORD_URL = "https://zenodo.org/records/22763120"

import download_dataset as dl
import importlib
dl = importlib.reload(dl)

info = dl.inspect_dataset()
print(info)

if dl.dataset_ready():
    print("Dataset already present — skip download.")
elif ZENODO_RECORD_URL.strip():
    dl.download_and_extract(ZENODO_RECORD_URL)
    print(dl.inspect_dataset())
else:
    print(
        "No dataset yet. Either set ZENODO_RECORD_URL above, or copy/symlink "
        "the BBBC021 sample folders into data/dataset/ "
        "(each folder must contain image.tif)."
    )
""")
md(cells, r"""
## 3. Sanity check one sample
""")
code(cells, r"""
from pathlib import Path
import tifffile

info = dl.inspect_dataset()
assert info["n_samples"] > 0, "data/dataset/ is empty"
ex = info["example"]
sample = Path(info["path"]) / ex["sample_id"]
img = tifffile.imread(sample / "image.tif")
print("example :", sample.name)
print("image   :", img.shape, img.dtype)
print("files   :", ex["files"])
print("slices  :", ex["has_slices"], " segmentation:", ex["has_segmentation"])
print("n samples:", info["n_samples"])
""")
write(nb, cells, "01_setup_environment_and_data.ipynb")

# ---------------------------------------------------------------------------
# 02 extract
# ---------------------------------------------------------------------------
nb, cells = new_nb()
md(cells, r"""
# 02 — Extract the 467 MorphAgent features from images

`source/feature_library/` holds one folder per feature:

- **438 code features** — `extract.py` with `extract(img, seg) → float`
- **29 VLM features** — a planner record scored by a vision model (0–100)

Images come from `data/dataset/` (notebook 01). This notebook runs a **smoke test** on a few samples. A full 3,552-image × 467-feature pass is documented but not launched here.
""")
code(cells, BOOTSTRAP + r"""
import pandas as pd
from IPython.display import display
import extract_morphagent_features as feat
feat = importlib.reload(feat)

manifest = pd.read_csv(P.FEATURE_LIBRARY / "manifest.csv")
display(manifest["method"].value_counts().to_frame("n"))
print("Library:", P.FEATURE_LIBRARY)
print("Dataset:", P.DATASET_DIR)
""")
md(cells, r"""
## 1. Code feature smoke

No API is required. The extractor loads `image.tif` plus `segmentation/*.tif`.
""")
code(cells, r"""
from IPython.display import display

ids = feat.list_sample_ids(feat.require_dataset())
print("n samples:", len(ids))
print("first 5 :", ids[:5])

code_feature = "tubulin_intensity_total"
code_path = P.FEATURE_LIBRARY / "code" / code_feature / "extract.py"
print("\n---", code_path, "---")
print(code_path.read_text(encoding="utf-8")[:1600])

code_df = feat.run_code_feature(code_feature, P.DATASET_DIR, ids[:5])
feat.OUTPUT_SMOKE.mkdir(parents=True, exist_ok=True)
code_df.to_csv(feat.OUTPUT_SMOKE / f"smoke_code_{code_feature}.csv", index=False)
display(code_df)
""")
md(cells, r"""
## 2. VLM credentials (your endpoint)

This tutorial **does not ship** a base URL or API key. Fill in an OpenAI-compatible vision endpoint, or leave the strings empty to skip the live VLM call.

Do not use `input()` / `getpass()` here — paste into the variables below and re-run the cell.
""")
code(cells, r"""
# Your OpenAI-compatible vision API (leave blank to skip VLM).
VLM_API_BASE_URL = ""   # e.g. "https://api.example.com/v1"
VLM_API_KEY = ""
VLM_MODEL = "gpt-4o"

if VLM_API_BASE_URL.strip() and VLM_API_KEY.strip():
    feat.configure_vlm_api(VLM_API_BASE_URL, VLM_API_KEY, VLM_MODEL)
else:
    print("VLM skipped: paste VLM_API_BASE_URL and VLM_API_KEY above to enable section 3.")
""")
md(cells, r"""
## 3. VLM feature smoke

One feature, two images. Inputs are the per-channel PNGs under `slices/`.
""")
code(cells, r"""
import json

vlm_feature = "vlm_nuclear_cap_presence"
spec = json.loads((P.FEATURE_LIBRARY / "vlm" / vlm_feature / "feature.json").read_text(encoding="utf-8"))
print("name       :", spec.get("name"))
print("category   :", spec.get("category"))
print("description:\n", spec.get("description"))

if not feat.vlm_api_ready():
    print("\nNo VLM credentials — live call skipped.")
    vlm_df = None
else:
    vlm_df = feat.run_vlm_feature(vlm_feature, P.DATASET_DIR, ids[:2])
    vlm_df.to_csv(feat.OUTPUT_SMOKE / f"smoke_vlm_{vlm_feature}.csv", index=False)
    display(vlm_df[["sample_id", vlm_feature, "error"]])
""")
md(cells, r"""
## 4. Full-dataset wall time (do not run here)

Smoke timings on this machine, extrapolated to 3,552 images:

| Workload | Approx. serial time |
|----------|---------------------|
| 438 code features | ~29 h (~2 h with 16 processes) |
| 29 VLM features, one call each | hundreds of hours |
| VLM batched (1 call / image) | ~14 h, depends on your API |

CLI:

```bash
python code/extract_morphagent_features.py --smoke
python code/extract_morphagent_features.py --smoke --api-base https://YOUR_HOST/v1 --api-key YOUR_KEY --api-model YOUR_MODEL
```
""")
write(nb, cells, "02_extract_morphagent_features.ipynb")

# ---------------------------------------------------------------------------
# 03 figures
# ---------------------------------------------------------------------------
nb, cells = new_nb()
md(cells, r"""
# 03 — Reproduce the main BBBC021 figures

This notebook replays the manuscript BBBC021 panels from **cached metrics** shipped in `source/cached_results/` (small JSON/CSV, no 400 MB evaluation tables).

1. Minimum non-redundant feature count (CellProfiler 1,284 / DeepProfiler 440 / MorphAgent 291).
2. Four-task benchmark: perturbation detection, MoA prediction, same-MoA matching, L1000 regression. Bars: CellProfiler, pretrained DeepProfiler, DeepProfiler retrained on BBBC021, MorphAgent. Significance marks follow the manuscript; `ns` is omitted.

Optional full recompute needs large evaluation CSVs under `data/evaluation/` (not on GitHub). Leave `RECOMPUTE = False`.
""")
code(cells, BOOTSTRAP + r"""
import pandas as pd
from IPython.display import Image, display
import reproduce_main_figures as bb
bb = importlib.reload(bb)

RECOMPUTE = False
bb.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Cached results:", P.CACHED_RESULTS)
print("Eval tables   :", bb.evaluation_tables_available())
print("Output dir    :", bb.OUTPUT_DIR)
""")
md(cells, r"""
## 1. MorphAgent 467-name vocabulary
""")
code(cells, r"""
names_467 = bb.load_morphagent_467_names()
print(f"{len(names_467)} named features")
for i, n in enumerate(names_467[:15], 1):
    print(f"  {i:3d}. {n}")
print("...")
print("vlm_ features:", sum(n.startswith("vlm_") for n in names_467))
""")
md(cells, r"""
## 2. Non-redundant feature count

A feature is redundant if Pearson $|r| > 0.9$ with an earlier column (greedy filter). Manuscript lock-in: **1,284 / 440 / 291**.
""")
code(cells, r"""
if RECOMPUTE and bb.evaluation_tables_available():
    counts = bb.compute_nonredundant_counts(recompute=True)
else:
    counts = bb.load_cached_nonredundant()
display(counts)
png = bb.plot_nonredundant(counts, out_stem=bb.OUTPUT_DIR / "min_nonredundant_subset")
display(Image(filename=str(png)))
for method, target in bb.PAPER_NONREDUNDANT.items():
    got = int(counts.loc[method, "n_min_subset"])
    print(f"  {method:13s}  reproduced={got:4d}  manuscript={target:4d}")
""")
md(cells, r"""
## 3. Four-task benchmark

Shared protocol: numeric columns, impute non-finite values, z-score. Uncertainty is a 95% bootstrap CI (2,000 resamples). The figure copies the manuscript brackets and hides non-significant comparisons.

L1000 $R^2$ uses the released random-split MLP run (not retrained here).
""")
code(cells, r"""
if RECOMPUTE and bb.evaluation_tables_available():
    datasets = bb.load_benchmark_tables(morphagent_source="paper_eval")
    results = bb.run_four_benchmarks(datasets, recompute=True, cache_tag="paper_eval")
else:
    results = bb.load_cached_four_benchmarks()

for task_key, title in [
    ("perturbation_detection", "Perturbation detection"),
    ("moa_detection", "MoA prediction"),
    ("same_moa_matching", "Same-MoA matching"),
    ("l1000_regression", "L1000 regression"),
]:
    task = results[task_key]
    print(f"\n{title}")
    for m in bb.METHODS_ORDER:
        r = task["methods"][m]
        print(
            f"  {bb.method_label(m):40s}  "
            f"{r['metric_name']}={r['metric']:.4f}  "
            f"CI95=[{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]"
        )
""")
md(cells, r"""
## 4. Four-panel figure
""")
code(cells, r"""
png = bb.plot_four_benchmarks(results, out_stem=bb.OUTPUT_DIR / "four_benchmarks")
display(Image(filename=str(png)))
cmp_df = bb.compare_to_paper(results)
display(cmp_df)
""")
md(cells, r"""
## 5. Supplementary: shared ANOVA ranking on perturbation detection
""")
code(cells, r"""
if RECOMPUTE and bb.evaluation_tables_available():
    datasets_named = bb.load_benchmark_tables(morphagent_source="467")
    si = bb.run_si_anova_perturbation_detection(datasets_named, recompute=True)
else:
    si = bb.load_cached_si_anova()
display(si.pivot(index="k", columns="method", values="mAP"))
png = bb.plot_si_anova_perturbation(si, out_stem=bb.OUTPUT_DIR / "si_anova_perturbation_detection")
display(Image(filename=str(png)))
""")
md(cells, r"""
## 6. Files

- `source/cached_results/` — metrics + figures shipped with the tutorial
- `data/outputs/` — plots written by this notebook
- `source/feature_library/` — the 467 extractors used in notebook 02
""")
write(nb, cells, "03_reproduce_main_figures.ipynb")
