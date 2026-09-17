#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUT_ROOT="${1:-$ROOT/outputs}"
mkdir -p "$OUT_ROOT"

"$PYTHON_BIN" "$ROOT/scripts/run_image_feature_young_old_pca12_selected.py" \
  "$ROOT/input/features_code_filtered_old_plus_20210624_Young.csv" \
  "$ROOT/input/selected_features.csv" \
  "$OUT_ROOT/image_morphagent_25" \
  --cv-splits 5 --cv-repeats 5 --random-state 42 \
  --pseudotime-csv "$ROOT/input/cell_pseudotime_selected_params.csv" \
  --pseudotime-palette "$ROOT/input/pseudotime_umap_strict_coolwarm_palette.txt"

"$PYTHON_BIN" "$ROOT/scripts/run_image_feature_young_old_pca12_search.py" \
  "$ROOT/input/omics_aligned_to_data_test_25_old_plus_20210624_Young.tsv" \
  "$OUT_ROOT/transcriptome_top500" \
  --top-variable-features 2000 --prefilter-top-n 2000 \
  --ranked-subsets --subset-sizes 500,1000,1500,2000 \
  --cv-splits 5 --cv-repeats 5 --random-state 42 \
  --pseudotime-csv "$ROOT/input/cell_pseudotime_selected_params.csv" \
  --pseudotime-palette "$ROOT/input/pseudotime_umap_strict_coolwarm_palette.txt"

"$PYTHON_BIN" "$ROOT/scripts/run_image_feature_young_old_pca12_search.py" \
  "$ROOT/input/features_manuel_old_plus_20210624_Young.csv" \
  "$OUT_ROOT/handcrafted_poly3" \
  --use-all-features --use-raw-two-features-as-pca \
  --raw-two-features-classifier poly3 \
  --pseudotime-csv "$ROOT/input/cell_pseudotime_selected_params.csv" \
  --random-state 42 --cv-splits 5 --cv-repeats 5

"$PYTHON_BIN" "$ROOT/scripts/run_image_feature_young_old_pca12_selected.py" \
  "$ROOT/input/merged_numeric_features_cleaned.csv" \
  "$ROOT/input/selected_features.csv" \
  "$OUT_ROOT/transfer_fixed25_validation162" \
  --cv-splits 5 --cv-repeats 5 --random-state 42

"$PYTHON_BIN" "$ROOT/scripts/run_image_feature_young_old_pca12_selected_blurred_grid.py" \
  --features-csv "$ROOT/input/merged_blurred_features.csv" \
  --selected-features-csv "$ROOT/input/selected_features.csv" \
  --out-dir "$OUT_ROOT/blurred_fixed25_grid" \
  --cv-splits 5 --cv-repeats 5 --random-state 42

echo "Results: $OUT_ROOT"
