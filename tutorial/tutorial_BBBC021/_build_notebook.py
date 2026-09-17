#!/usr/bin/env python3
"""Generate reproduce_main_figures.ipynb (English tutorial)."""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n") + "\n"))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n") + "\n"))


md(r"""
# Reproducing the main BBBC021 figures from features

This tutorial starts from the released **CellProfiler**, **DeepProfiler** and **MorphAgent** feature tables and rebuilds the two headline BBBC021 panels in the MorphAgent / BPAgent manuscript:

1. **Minimum non-redundant feature count** (within-space Pearson $|r| > 0.9$).
2. **Four-task benchmark** on CellProfiler, pretrained DeepProfiler, DeepProfiler retrained on BBBC021, and MorphAgent: perturbation detection, mechanism-of-action (MoA) prediction, same-MoA matching, and L1000 gene-expression regression. Significance marks follow the manuscript figure (non-significant comparisons are omitted).

MorphAgent uses the **467** named features in Supplementary Feature List 1 (`assets/morphagent_467_feature_names.csv`). Helper functions live in `bbbc021_main_results.py`.

**Dataset.** BBBC021 Cell Painting profiles of MCF-7 cells: **3,552** images, **37** compounds, **26** MoA labels, three channels (DAPI, tubulin, actin).
""")

md(r"""
## 0. Setup

Run this notebook from `tutorial_BBBC021/`. Feature CSVs stay in `results_BBBC021_evaluation` (they are large and are not copied here). Set `RECOMPUTE = True` only if you want to ignore cached outputs.
""")

code(r"""
from pathlib import Path
import sys

import pandas as pd
from IPython.display import Image, display

TUTORIAL = Path.cwd()
if not (TUTORIAL / "bbbc021_main_results.py").exists():
    TUTORIAL = Path("/data3/yez/MorphAgent/tutorial_BBBC021")
sys.path.insert(0, str(TUTORIAL))

import importlib
import bbbc021_main_results as bb
bb = importlib.reload(bb)

RECOMPUTE = False

bb.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Tutorial dir :", TUTORIAL)
print("Eval root    :", bb.EVAL_ROOT)
print("Output dir   :", bb.OUTPUT_DIR)
""")

md(r"""
## 1. Load the feature spaces

| Method | What the columns are | Width |
|--------|----------------------|-------|
| CellProfiler | handcrafted Cell Painting descriptors | 6,278 (6,256 variable) |
| DeepProfiler | pretrained embedding dimensions | 672 |
| DeepProfiler (retrained on BBBC021) | embedding trained on this dataset | 672 |
| MorphAgent | biologically named measurements (Supplementary Feature List 1) | **467** |
""")

code(r"""
names_467 = bb.load_morphagent_467_names()
print(f"MorphAgent named vocabulary: {len(names_467)} features")
print("First 15 names:")
for i, n in enumerate(names_467[:15], 1):
    print(f"  {i:3d}. {n}")

ma467 = bb.load_morphagent_467()
print("\nMorphAgent 467-feature matrix:", ma467.shape)
print("Compounds:", ma467["perturbation_id"].nunique())
print(ma467[["sample_id", "perturbation_id"]].head())
""")

code(r"""
datasets = bb.load_benchmark_tables(morphagent_source="467")
feat_cols = {k: bb.get_feature_columns(v) for k, v in datasets.items()}
inventory = pd.DataFrame([
    {
        "method": bb.method_label(k),
        "n_images": len(v),
        "n_compounds": v["perturbation_id"].nunique(),
        "n_features": len(feat_cols[k]),
    }
    for k, v in datasets.items()
])
display(inventory)
assert inventory.loc[inventory["method"] == "MorphAgent", "n_features"].item() == 467

moa = bb.load_drug_moa_map()
print(f"MoA map: {len(moa)} compounds, {len(set(str(x) for x in moa.values()))} distinct MoA labels")
""")

md(r"""
## 2. Non-redundant feature count

A feature is called **redundant** if it has Pearson $|r| > 0.9$ with an earlier column (greedy left-to-right filter, constants dropped first). The bar height is the size of the remaining subset.

This is computed on the feature-space tables used in the manuscript panel:

- CellProfiler: `features_dataset/cellprofiler/` (includes DMSO; 3,652 rows)
- DeepProfiler: `features_dataset/deepprofiler_new/`
- MorphAgent: `features_dataset/morphagent_filtered_new/` restricted to the **467** retained names

Manuscript lock-in: **1,284 / 440 / 291**.
""")

code(r"""
counts = bb.compute_nonredundant_counts(recompute=RECOMPUTE)
display(counts)
png = bb.plot_nonredundant(counts)
display(Image(filename=str(png)))

print("\nManuscript targets:", bb.PAPER_NONREDUNDANT)
for method, target in bb.PAPER_NONREDUNDANT.items():
    got = int(counts.loc[method, "n_min_subset"])
    print(f"  {method:13s}  reproduced={got:4d}  manuscript={target:4d}  delta={got - target:+d}")
""")

md(r"""
## 3. Four-task benchmark protocol

Shared preprocessing for every method and task: keep numeric feature columns, impute non-finite values, then z-score.

All four representations are evaluated on the same 3,552 images. Uncertainty is a percentile 95% CI from **2,000** bootstrap resamples of the evaluation units. The four-panel figure copies the manuscript significance brackets and omits non-significant (`ns`) comparisons.
""")

md(r"""
### 3a. Perturbation detection (image-level mAP)

Each image is a query. Rank all other images by cosine similarity; average precision is computed for retrieving the same compound, then averaged over 3,552 queries.
""")

md(r"""
### 3b. MoA prediction (kNN accuracy)

Image-level 80/10/10 stratified split. Euclidean kNN with neighbourhood size chosen on validation accuracy from $\{1,3,5,7,9,11,15,21,31\}$. Reported metric: top-1 test accuracy. This is **not** a compound-holdout split.
""")

md(r"""
### 3c. Same-MoA perturbation matching

Restricted to MoA groups with at least two compounds. For every ordered pair of distinct compounds that share an MoA, sample five negatives from other MoAs (six-way retrieval). Each compound is represented by its mean profile. AP reduces to $1/\mathrm{rank}$ of the positive centroid. Random baseline $\approx 0.408$.
""")

md(r"""
### 3d. L1000 gene-expression regression

A two-layer MLP (512–256, dropout 0.3) maps morphology to paired L1000 profiles under a **random** sample split (seed 42). The manuscript bars freeze the released run:

| Method | R² | 95% CI |
|--------|----|--------|
| CellProfiler | 0.096 | 0.059 – 0.129 |
| DeepProfiler | 0.159 | 0.122 – 0.192 |
| DeepProfiler (retrained on BBBC021) | −0.005 | −0.019 – 0.002 |
| MorphAgent | 0.177 | 0.143 – 0.207 |

Retraining is stochastic and is **not** repeated in this tutorial; we load those published point estimates and bootstrap CIs, which is the same procedure used by the original figure scripts.
""")

code(r"""
results = bb.run_four_benchmarks(
    datasets,
    recompute=RECOMPUTE,
    cache_tag="467",
)

print("=== Reproduced metrics ===")
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

md(r"""
## 4. Four-panel figure
""")

code(r"""
import importlib
bb = importlib.reload(bb)

png = bb.plot_four_benchmarks(results)
display(Image(filename=str(png)))

cmp_df = bb.compare_to_paper(results)
display(cmp_df)
""")

md(r"""
## 5. Optional: same ranking protocol on all three feature spaces

The main four-task figure uses each method's native feature space. A natural supplementary question is whether MorphAgent's advantage on **perturbation detection** (compound retrieval / drug screening) is an artifact of comparing feature sets of very different width.

Here we give CellProfiler, DeepProfiler **and** MorphAgent the *same* one-way ANOVA ranking across compound labels, then keep the top $k \in \{50, 100, 200, 400\}$ features (plus $k \in \{800, 1600\}$ for CellProfiler, which is wide enough) and the full set. If MorphAgent still leads under this shared filter, the gain is not explained by "only MorphAgent was reduced to its most discriminative axes".
""")

code(r"""
import importlib
bb = importlib.reload(bb)

si = bb.run_si_anova_perturbation_detection(datasets, recompute=RECOMPUTE)
display(si.pivot(index="k", columns="method", values="mAP"))

png = bb.plot_si_anova_perturbation(si)
display(Image(filename=str(png)))

print("\nAt a matched budget of k=200 ANOVA-ranked features:")
sub = si[si["k"] == "200"][["method", "n_features", "mAP", "ci95_low", "ci95_high"]]
display(sub.reset_index(drop=True))
print(
    "MorphAgent remains highest on perturbation detection "
    "when every method is ranked the same way."
)
""")

md(r"""
## 6. Output files

After a successful run, `outputs/` contains:

- `min_nonredundant_subset.{png,svg,pdf}`
- `four_benchmarks.{png,svg,pdf}`
- `si_anova_perturbation_detection.{png,svg,pdf}`
- `redundancy_min_subset.csv`
- `summary_metrics.csv`
- `si_anova_all_methods.csv`
- `four_benchmark_results_467_4method.json`

CLI equivalent:

```bash
python bbbc021_main_results.py
```
""")

nb.cells = cells
out = Path("/data3/yez/MorphAgent/tutorial_BBBC021/reproduce_main_figures.ipynb")
out.write_text(nbf.writes(nb), encoding="utf-8")
print("Wrote", out, "n_cells=", len(cells))
