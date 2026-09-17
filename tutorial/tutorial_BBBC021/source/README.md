# source/

Everything the notebooks need besides the images.

| Path | Role |
|------|------|
| `feature_library/` | 467 MorphAgent features: 439 `code/<name>/extract.py` and 28 `vlm/<name>/feature.json` |
| `feature_library/manifest.csv` | Name, method, category, relative path, description |
| `cached_results/` | Cached metrics + PNG/SVG used by notebook 03 |
| `morphagent_467_feature_names.csv` | Canonical vocabulary (n = 467) |
| `drug_moa_source.csv` | Compound → mechanism-of-action map |
| `l1000_per_method.csv`, `l1000_pairwise.csv` | Published L1000 MLP random-split $R^2$ |
| `vlm_scoring.json` | Shared VLM scoring prompt template |

Code features are self-contained Python (`extract(img, seg) → float`). They import only `numpy`, `scipy`, and `scikit-image`. VLM features are scored with `vlm_scoring.json` against the PNG slices of each sample.
