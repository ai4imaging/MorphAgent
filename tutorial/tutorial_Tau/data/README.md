# data/

## `tables/`

Already-measured features, bundled with the tutorial (~15 MB). No raw microscopy.

| File | Contents |
|---|---|
| `wt_morphagent_301_features.csv` | 301 MorphAgent features, paired WT Tau discovery cohort (58 cells) |
| `wt_expert_16_features.csv` | 16 expert-designed features, same cohort |
| `wt_mean_cell_tau_intensity.csv` | mean Tau intensity within each cell mask |
| `mutant_morphagent_features.csv` | MorphAgent features, mutant cohort, super-resolution and wide-field |
| `mutant_expert_features.csv` | expert-designed features, same cohort |
| `predict_morphagent_400_features.csv` | the 400 prediction features |
| `predict_expert_features.csv` | expert-designed features for the prediction cohort |
| `predict_transcriptome.csv` | paired gene expression matrix |

Imaging modality (`sr_` / `wf_`) and genotype are encoded in the cell identifier.

## `outputs/`

Figures and CSVs written by `notebook/reproduce_tau_experiments.ipynb`. Regenerated on every run; not shipped.
