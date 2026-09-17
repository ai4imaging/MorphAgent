# MorphAgent HSC tutorial — Figure 3

One notebook. It replays the Young/Old PCA panels from:

- `HSC_Fig3a_YoungOld_reproduction_20260916.zip` (110-cell discovery)
- `HSC_Fig3a_YoungOld_reproduction_20260916 2` (162-cell confocal transfer and blur)

Copied under `source/fig3a_bundle/`.

```
tutorial_HSC/
  notebook/hsc.ipynb
  source/fig3a_bundle/
    input/selected_features.csv              locked 25 names
    input/features_code_filtered_*.csv       110-cell MorphAgent features
    input/features_manuel_*.csv              110-cell handcrafted features
    input/omics_aligned_*.tsv                paired transcriptome
    input/merged_numeric_features_cleaned.csv  162-cell confocal transfer
    input/merged_blurred_features.csv          same 162 cells, blurred
    scripts/
  data/outputs/
```

## Setup

```bash
python -m pip install -r requirements.txt
```

Open `notebook/hsc.ipynb` and run all cells. Same functions as `source/fig3a_bundle/run_all.sh` (blur panel uses `--combo-index 11`, the published 0.795 setting).

## Expected CV ROC AUC (5-fold × 5 repeats, seed 42)

| Analysis | n | Paper | This notebook |
|----------|--:|------:|--------------:|
| MorphAgent, fixed 25 image features | 110 | 0.739 | 0.739 |
| Handcrafted, two features, polynomial degree 3 | 110 | 0.575 | 0.575 |
| Transcriptome, top 500 of ranked 2,000 | 110 | 0.920 | 0.920 |
| Transfer, same 25 features, confocal | 162 | 0.905 | 0.905 |
| Blurred 162 cells, combo 011 | 162 | 0.795 | 0.795 |

The scatter plot is fitted on all cells in that panel; its in-sample `full_roc_auc` is not the reported CV AUC.

The 25-name list is taken as a given. All 25 names are columns of the discovery and transfer tables. This notebook evaluates that list; it does not re-select features.

Combo 011 is `standard` scaler, clip quantile 0.02, signed log1p, no PCA whitening. Running the full 36-combination blur grid yields a higher best AUC (~0.835); that is not the published panel.

## CLI

```bash
PYTHON_BIN=python bash source/fig3a_bundle/run_all.sh data/outputs
```
