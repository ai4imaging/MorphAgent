# MorphAgent Tau tutorial

MorphAgent designs image features for a biological question rather than reusing
a fixed descriptor panel. `notebook/reproduce_tau_experiments.ipynb` asks what
those features buy you on Tau, and reproduces the published panels end to end.

Everything needed is in this folder: no downloads, no credentials, no external
checkouts. The bundled measurement tables total about 15 MB.

Recorded walkthrough: [`tutorial_Tau.mp4`](tutorial_Tau.mp4) (2½ min) runs the notebook
end to end, including the gene-prediction section that takes several minutes.

## Quick start

```bash
pip install -r requirements.txt
jupyter lab notebook/reproduce_tau_experiments.ipynb
```

Run the notebook top to bottom. Sections 1–4 and 6–7 take well under a minute;
section 5 scores the full transcriptome and takes about seven minutes on 24
workers, with a `MAX_GENES` switch for a fast pass.

## What is reproduced

| Section | Panel | What it establishes about MorphAgent features |
|---|---|---|
| 2 | Tau feature space (PCA of the paired WT discovery cohort) | they span a far richer phenotypic space than an expert panel, while preserving its structure |
| 3 | WT versus P301S+S320F, super-resolution and wide-field | they separate genotypes better in both modalities, most of all at super-resolution |
| 4 | Six-class Tau genotype classification | they resolve six genotypes at higher accuracy and lower macro false-positive rate |
| 5 | Gene expression prediction from morphology | they carry transcriptomic information that Tau abundance alone does not |
| 6 | Top-level morphology–GO heatmap | morphology-defined biological processes and GO programs converge on the same genes |
| 7 | Middle-level architecture–GRN heatmap | those correspondences resolve into subcellular architecture × GRN modules |

Each section recomputes its figure from `data/tables/` and then prints the
recomputed value next to the published one, so the agreement is visible rather
than asserted. The classification panels match the manuscript to three decimal
places on all sixteen reported numbers. In the prediction panel the MorphAgent
arm reproduces the published per-gene scores exactly; the expert and
Tau-intensity arms draw regressor configurations per gene from a seeded
generator and land within about 0.002 of the published means.

Each analysis module also runs standalone from the tutorial root:

```bash
python code/tau_pca.py                              # feature-space panel
python code/tau_classification.py                   # both classification panels
python code/tau_classification.py --only binary     # just the binary panel
python code/tau_prediction.py --n-jobs 24           # prediction panel, full transcriptome
python code/tau_prediction.py --max-genes 800       # prediction panel, quick pass
python code/tau_heatmap.py                          # top- and middle-level heatmaps
python code/build_notebook.py                       # regenerate the notebook
```

## Layout

```
tutorial_Tau/
  notebook/reproduce_tau_experiments.ipynb  the tutorial
  code/                        analysis modules (paths, PCA, classification, prediction, heatmap)
  source/
    feature_lists/             the two MorphAgent feature lists
    splits/                    ten fixed train/validation/test resamples
    cached_results/            the values reported in the manuscript
    heatmap/                   morphology classes, GO terms, architectures, GRN modules
  data/
    tables/                    bundled measurement tables
    outputs/                   figures and CSVs written by the notebook
  requirements.txt
```

## The two feature lists

`source/feature_lists/` ships both MorphAgent feature sets as plain CSVs.
Feature names are keyed to the column names in `data/tables/`, so any feature
named in a list can be looked up directly in the corresponding table.

| List | Size | Composition | Used by |
|---|---|---|---|
| `feature_list_301_classification.csv` | 301 | code-derived descriptors, retained after review with semantically overlapping features collapsed to one representative each | the classification figure's feature space |
| `feature_list_400_prediction.csv` | 400 | 200 code-derived (`code_` prefix) plus 200 vision-language (`vlm_` prefix) | transcriptome prediction |

The two lists were designed on different cohorts and are independent of one
another. The tutorial ships feature names and their measured values; the
descriptor generation prompts are not included.

## Data

`data/tables/` contains already-measured features rather than raw microscopy,
which is what keeps the tutorial self-contained.

| Table | Contents |
|---|---|
| `wt_morphagent_301_features.csv` | 301 MorphAgent features, paired WT Tau discovery cohort (58 cells) |
| `wt_expert_16_features.csv` | 16 expert-designed features, same cohort |
| `wt_mean_cell_tau_intensity.csv` | mean Tau intensity within each cell mask |
| `mutant_morphagent_features.csv` | MorphAgent features, mutant cohort, super-resolution and wide-field |
| `mutant_expert_features.csv` | expert-designed features, same cohort |
| `predict_morphagent_400_features.csv` | the 400 prediction features |
| `predict_expert_features.csv` | expert-designed features for the prediction cohort |
| `predict_transcriptome.csv` | paired gene expression matrix |

Imaging modality and genotype are encoded in the cell identifier (`sr_` / `wf_`
prefix, genotype in the body), so the classification sections derive their
labels rather than reading them from a separate annotation file.

## Methods in brief

**Feature space.** Each space is z-scored and projected onto its own first two
principal components. Point colour and size encode mean cellular Tau intensity
on a log scale. Grey ellipses are 95% covariance ellipses of the low- and
high-intensity halves of the cohort, included as a visual guide rather than as a
fitted result.

**Classification.** Median imputation, z-scoring, and balanced logistic
regression under five-fold stratified cross-validation. For the binary task the
expert panel uses all of its features while MorphAgent is narrowed to the 50
descriptors with the highest single-feature cross-validated AUC on the
super-resolution subset, applied unchanged to both modalities. The six-class
task uses every available feature in each family. Significance comes from a
paired stratified bootstrap over out-of-fold predictions: each cell is predicted
once in its held-out fold, cells are resampled within genotype, and both arms
are scored on the same resampled cells. Comparisons are reported in a table
rather than annotated on the six-class bars.

**Prediction.** Ten fixed resamples ship in `source/splits`. Each arm is
evaluated with the procedure used for it in the manuscript, and those procedures
differ; `code/tau_prediction.py` records which applies to which arm via
`ARM_PROTOCOL`. The MorphAgent arm ranks predictors by absolute Pearson
correlation on train and validation cells, fits a support-vector regressor on the
top *k* for *k* in {1, 2, 4, 8, 16, 32, 64}, and averages the two best-scoring
*k*. The expert and Tau-intensity arms use all predictors of the arm with no
ranking step, selecting one of ten regressor configurations by Spearman
correlation on inner validation cells. Bars report the mean across genes, each
arm ranked by its own scores; a gene whose correlation is undefined counts as
zero so all three panels average over the same gene universe.

**Hierarchy heatmaps.** Eleven morphology-defined biological processes are
assigned the union of genes correlated with their member features. Each enriched
GO term contributes its member genes (nine mitochondrial terms merged into one
column). The top-level matrix is the shared-gene count between every process and
every GO term. Two boxed pairs are then expanded at the middle level:
subcellular architectures versus GRN modules (TF plus top-decile targets)
inside ribonucleoprotein complex biogenesis and the spliceosomal complex.

## Requirements

Python 3.9 or newer plus the packages in `requirements.txt`. Section 1 of the
notebook checks the environment and reports anything missing.
