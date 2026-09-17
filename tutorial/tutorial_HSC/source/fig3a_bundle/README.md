# HSC Young/Old classification: Figure 3 reproduction

This bundle contains the code and input files used by five supplied commands. It reproduces the reported cross-validation AUCs for MorphAgent image features, transcriptomics, handcrafted features and the fixed-25 validation analysis. It also runs the blurred-feature preprocessing grid. No original result directories are modified.

## Run

Use Python 3.10 (the source environment used Python 3.10.10). From this folder:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHON_BIN="$PWD/.venv/bin/python" bash run_all.sh
```

The `run_all.sh` command can also take an output directory, for example `PYTHON_BIN="$PWD/.venv/bin/python" bash run_all.sh /path/to/output`. The input paths in `run_all.sh` are relative to this folder, so the bundle can be moved to another computer. Installing the dependencies requires access to the packages or a compatible local package cache; the Python environment itself is not included.

To run only the two added analyses after setting up `.venv`, use:

```bash
.venv/bin/python scripts/run_image_feature_young_old_pca12_selected.py \
  input/merged_numeric_features_cleaned.csv \
  input/selected_features.csv \
  outputs/transfer_fixed25_validation162 \
  --cv-splits 5 --cv-repeats 5 --random-state 42

.venv/bin/python scripts/run_image_feature_young_old_pca12_selected_blurred_grid.py \
  --features-csv input/merged_blurred_features.csv \
  --selected-features-csv input/selected_features.csv \
  --out-dir outputs/blurred_fixed25_grid \
  --cv-splits 5 --cv-repeats 5 --random-state 42
```

Run these from the bundle root. The extra `--features-csv`, `--selected-features-csv` and `--out-dir` options on the blur-grid command replace the script's machine-specific default paths.

To regenerate only the blurred plot labeled `combo_011` (CV AUC 0.7946997301, displayed as 0.795), retain the original 36-combination numbering with `--combo-index 11`:

```bash
.venv/bin/python scripts/run_image_feature_young_old_pca12_selected_blurred_grid.py \
  --features-csv input/merged_blurred_features.csv \
  --selected-features-csv input/selected_features.csv \
  --out-dir outputs/blurred_combo011_only \
  --combo-index 11 \
  --cv-splits 5 --cv-repeats 5 --random-state 42
```

The SVG is `outputs/blurred_combo011_only/combo_011__scaler_standard__clip_0.02__log1p_1__whiten_0/pca12_group_scatter.svg`. Without `--combo-index`, the script still runs all 36 combinations.

## Inputs and code

- `scripts/run_image_feature_young_old_pca12_search.py`: common preprocessing, evaluation and plotting code; also runs the transcriptomic and handcrafted analyses.
- `scripts/run_image_feature_young_old_pca12_selected.py`: evaluates the fixed 25-feature image panel; imports the common script above.
- `scripts/run_image_feature_young_old_pca12_selected_blurred_grid.py`: evaluates the blurred 162-cell feature table over 36 preprocessing combinations; imports both scripts above.
- `input/features_code_filtered_old_plus_20210624_Young.csv`: image-derived feature table for 110 discovery cells.
- `input/selected_features.csv`: the fixed 25-feature list used for the image result.
- `input/merged_numeric_features_cleaned.csv`: unblurred image features for 162 validation cells.
- `input/merged_blurred_features.csv`: blurred image features for the same validation cohort.
- `input/omics_aligned_to_data_test_25_old_plus_20210624_Young.tsv`: paired transcriptomic matrix.
- `input/features_manuel_old_plus_20210624_Young.csv`: two handcrafted features.
- `input/cell_pseudotime_selected_params.csv`: cell pseudotime values used only for the pseudotime-colored plots.
- `input/pseudotime_umap_strict_coolwarm_palette.txt`: plot colors used by the first two commands.

## Expected results

| Analysis | Output table | CV ROC AUC |
| --- | --- | ---: |
| MorphAgent, fixed 25 image features | `outputs/image_morphagent_25/pca12_summary.csv` | 0.7386965812 (0.739) |
| Transcriptomics, top 500 of ranked 2,000 | `outputs/transcriptome_top500/pca12_search_summary.csv`, row `subset_size=500` | 0.9203937729 (0.920) |
| Handcrafted, two features, polynomial degree 3 | `outputs/handcrafted_poly3/pca12_search_summary.csv` | 0.5747710623 (0.575) |
| Fixed 25 image features on 162 validation cells | `outputs/transfer_fixed25_validation162/pca12_summary.csv` | 0.9046383266 (0.905) |
| Blurred 162-cell preprocessing grid, highest-ranked combination | `outputs/blurred_fixed25_grid/grid_summary.csv`, first row | 0.8349480432 (0.835) |

The unblurred analyses also include a Young/Old scatter plot as `pca12_group_scatter.svg` in their `top_500_features` or `all_features` subfolder, or directly in the fixed-25 output directories. The blur-grid analysis saves a plot for each preprocessing combination. The AUC is the mean of 5-fold stratified cross-validation repeated five times with random seed 42. Scatter plots are fitted on all cells in the corresponding dataset; their `full_roc_auc` is not the reported CV AUC.

## Provenance and interpretation

- The image command takes the 25-feature list as an existing input. That list was copied from a separate feature-search result containing 162 cells. This bundle reproduces the 0.739 evaluation on 110 cells, **not** the upstream selection of the 25 features. The feature-selection provenance should be reconciled with any Methods text claiming that the list was selected on the 110-cell discovery set.
- The `0.905` validation command reuses this exact 25-feature list without searching for another panel, but the provenance caveat above still applies.
- The blurred grid keeps the 25 features fixed while selecting the highest-scoring preprocessing combination on the blurred dataset. Its best AUC is a grid-search result, not an AUC from a preprocessing configuration fixed before the grid search.
- The transcriptomic command first retains the 2,000 genes with highest expression variance, then ranks those genes by label-based ANOVA F-statistic and evaluates the top 500 in that order. It does **not** select the 500 highest-variance genes directly. The ranking is computed before cross-validation, so the reported CV AUC is not an estimate from a fully nested feature-selection protocol.
- The handcrafted `0.575` value uses a degree-3 polynomial expansion followed by regularized logistic regression. Omitting `--raw-two-features-classifier poly3` changes the model to the default linear classifier and gives approximately 0.406, not 0.575.
- Pseudotime and palette inputs affect only the additional plot, not classification AUC.
