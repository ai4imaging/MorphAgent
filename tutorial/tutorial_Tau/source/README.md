# source/

Non-image assets the notebook needs besides `data/tables/`.

## `feature_lists/`

Two MorphAgent feature sets, as plain CSVs keyed to column names in `data/tables/`.

| File | Size | Used by |
|---|---|---|
| `feature_list_301_classification.csv` | 237 code + 64 vlm descriptors | the Tau feature-space panel |
| `feature_list_400_prediction.csv` | 200 `code_` + 200 `vlm_` | transcriptome prediction |

The lists were designed on different cohorts and are independent. Descriptor generation prompts are not included.

## `splits/`

Ten fixed train / validation / test resamples for the prediction panel (`repeat_000.json` … `repeat_009.json`). Index arrays only; no sample identifiers.

## `cached_results/`

The values reported in the manuscript. The notebook recomputes every figure from `data/tables/` and then prints the recomputed number next to these, so the agreement is visible rather than asserted.

## `heatmap/`

Inputs for the morphology–gene hierarchy panels.

| File | Contents |
|---|---|
| `morphology_classes.csv` | 11 morphology-defined biological processes, member features and genes |
| `go_terms.csv` | enriched GO terms (mitochondrial terms already merged) |
| `architectures.csv` | subcellular architectures nested under each process |
| `feature_genes.csv` | feature → correlated gene symbols |
| `grn_modules.csv` | GRN modules for ribonucleoprotein complex biogenesis and the spliceosomal complex |
