#!/usr/bin/env python3
"""Regenerate the tutorial notebook.

One notebook covers the Tau panels plus the morphology–gene heatmaps.
Kept in the repository so it can be rebuilt from a single source of truth
after the analysis modules change:

    python code/build_notebook.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from paths import ROOT

NOTEBOOK_NAME = "notebook/reproduce_tau_experiments.ipynb"


def md(cells: list, text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))


def code(cells: list, text: str) -> None:
    cells.append(nbf.v4.new_code_cell(dedent(text).strip()))


def build() -> Path:
    cells: list = []

    # ---------------------------------------------------------------- intro
    md(cells, """
    # Reproducing the MorphAgent Tau figures

    MorphAgent designs image features for a biological question, rather than
    reusing a fixed descriptor panel. This notebook asks what those features buy
    you on Tau, and reproduces the published panels end to end:

    | Section | Panel | What it establishes about MorphAgent features |
    |---|---|---|
    | 2 | Tau feature space | they span a far richer phenotypic space than an expert panel, while preserving its structure |
    | 3 | WT versus P301S+S320F | they separate genotypes better in both imaging modalities, most of all at super-resolution |
    | 4 | Six-class genotype | they resolve six Tau genotypes at higher accuracy and lower false-positive rate |
    | 5 | Gene expression prediction | they carry transcriptomic information that Tau abundance alone does not |
    | 6 | Top-level heatmap | morphology-defined biological processes and GO programs converge on the same genes |
    | 7 | Middle-level heatmap | those correspondences resolve into subcellular architecture × GRN modules |

    Everything needed is inside this folder — no downloads, no credentials, no
    external checkouts. Each section recomputes its figure from
    `data/tables/` or `source/heatmap/` and then prints the recomputed number
    next to the published one, so the agreement is visible rather than asserted.

    **Runtime.** Sections 1–4 and 6–7 take well under a minute in total. Section 5
    scores the full transcriptome and takes about seven minutes on 24 workers;
    it has a switch for a fast pass.
    """)

    # ---------------------------------------------------------------- setup
    md(cells, """
    ---
    ## 1. Setup, data and the two feature lists

    Install the dependencies once, from the tutorial root:

    ```bash
    pip install -r requirements.txt
    ```
    """)
    code(cells, """
    import sys
    from pathlib import Path

    def tutorial_root() -> Path:
        candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
        try:
            candidates = [Path(__vsc_ipynb_file__).resolve().parent, *candidates]
        except NameError:
            pass
        for p in candidates:
            if (p / "code" / "paths.py").is_file():
                return p
            if p.name == "notebook" and (p.parent / "code" / "paths.py").is_file():
                return p.parent
        raise FileNotFoundError("Cannot find the tutorial root (the folder that contains code/paths.py).")

    ROOT = tutorial_root()
    sys.path.insert(0, str(ROOT / "code"))
    print("Tutorial root:", ROOT)
    """)
    code(cells, """
    import importlib

    missing = []
    for module, package in {
        "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
        "sklearn": "scikit-learn", "matplotlib": "matplotlib", "joblib": "joblib",
    }.items():
        try:
            mod = importlib.import_module(module)
            print(f"  {package:<14} {getattr(mod, '__version__', 'unknown')}")
        except ImportError:
            missing.append(package)

    if missing:
        raise SystemExit("Missing packages: pip install " + " ".join(missing))
    print("\\nEnvironment ready.")
    """)
    code(cells, """
    import sys
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import Image, display
    import importlib

    def tutorial_root() -> Path:
        candidates = [Path.cwd().resolve(), *Path.cwd().resolve().parents]
        try:
            candidates = [Path(__vsc_ipynb_file__).resolve().parent, *candidates]
        except NameError:
            pass
        for p in candidates:
            if (p / "code" / "paths.py").is_file():
                return p
            if p.name == "notebook" and (p.parent / "code" / "paths.py").is_file():
                return p.parent
        raise FileNotFoundError("Cannot find the tutorial root (the folder that contains code/paths.py).")

    ROOT = tutorial_root()
    code_dir = str(ROOT / "code")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)

    import paths
    import tau_pca
    import tau_classification as tc
    import tau_prediction as tp
    import tau_heatmap as th
    importlib.reload(paths)
    importlib.reload(tau_pca)
    importlib.reload(tc)
    importlib.reload(tp)
    importlib.reload(th)

    %matplotlib inline
    pd.set_option("display.width", 220)

    # Collects every reproduced-versus-published comparison for the closing table.
    checks = {}
    print("Imported analysis modules from", code_dir)
    """)

    md(cells, """
    ### 1.1 What ships with the tutorial

    ```
    tutorial_Tau/
      notebook/                    this notebook
      code/                        the analysis modules it calls
      source/
        feature_lists/             the two MorphAgent feature lists
        splits/                    ten fixed train/validation/test resamples
        cached_results/            the values reported in the manuscript
        heatmap/                   morphology classes, GO terms, architectures, GRNs
      data/
        tables/                    the measurement tables (all bundled)
        outputs/                   figures and CSVs written by this notebook
    ```

    `data/tables/` holds already-measured features rather than raw microscopy,
    which is what keeps the tutorial self-contained and quick to run.
    """)
    code(cells, """
    rows = []
    for label, path in [
        ("§2  MorphAgent 301 features (WT cohort)", paths.WT_MORPHAGENT),
        ("§2  expert 16 features (WT cohort)", paths.WT_EXPERT),
        ("§2  mean cellular Tau intensity", paths.WT_TAU_INTENSITY),
        ("§3–4  MorphAgent features (mutant cohort)", paths.MUTANT_MORPHAGENT),
        ("§3–4  expert features (mutant cohort)", paths.MUTANT_EXPERT),
        ("§5  MorphAgent 400 features", paths.PRED_MORPHAGENT),
        ("§5  expert features", paths.PRED_EXPERT),
        ("§5  transcriptome", paths.PRED_TRANSCRIPTOME),
        ("§6–7  morphology classes", paths.HEATMAP_MORPHOLOGY_CLASSES),
        ("§6–7  GO terms", paths.HEATMAP_GO_TERMS),
        ("§7  subcellular architectures", paths.HEATMAP_ARCHITECTURES),
        ("§7  feature–gene map", paths.HEATMAP_FEATURE_GENES),
        ("§7  GRN modules", paths.HEATMAP_GRN_MODULES),
    ]:
        head = pd.read_csv(path, nrows=1)
        rows.append({
            "table": label,
            "file": path.name,
            "MB": round(path.stat().st_size / 1e6, 2),
            "columns": head.shape[1],
            "rows": sum(1 for _ in open(path, encoding="utf-8")) - 1,
        })

    inventory = pd.DataFrame(rows)
    print(inventory.to_string(index=False))
    print(f"\\nTotal bundled: {inventory.MB.sum():.1f} MB")
    """)

    md(cells, """
    ### 1.2 The two feature lists

    Two MorphAgent feature sets appear in this tutorial, both shipped as plain
    CSVs under `source/feature_lists/`:

    * **301 features** — the Tau descriptor space behind the classification
      figure: 237 code-derived measurements plus 64 vision-language
      descriptors, retained after review of a larger candidate set with
      semantically overlapping descriptors collapsed to one representative
      each.
    * **400 features** — the descriptor space used for transcriptome
      prediction: 200 code-derived measurements (`code_` prefix) plus 200
      vision-language descriptors (`vlm_` prefix).

    The two lists were designed on different cohorts and are independent of one
    another. The tutorial ships the feature names and their measured values; the
    descriptor generation prompts are not included.
    """)
    code(cells, """
    list_301 = pd.read_csv(paths.LIST_301)
    list_400 = pd.read_csv(paths.LIST_400)

    print(f"301-feature list: {len(list_301)} features")
    print(list_301.feature_kind.value_counts().to_string())
    print("review score distribution:")
    print(list_301.review_score.value_counts().sort_index().to_string())

    print(f"\\n400-feature list: {len(list_400)} features")
    print(list_400.feature_kind.value_counts().to_string())
    """)
    code(cells, """
    print("301-feature list — code-derived and vision-language examples")
    print(list_301.groupby("feature_kind").head(5).to_string(index=False))

    print("\\n400-feature list — code-derived and vision-language examples")
    print(list_400.groupby("feature_kind").head(4).to_string(index=False))
    """)

    md(cells, """
    Both lists are keyed to the column names in `data/tables/`, so every feature
    named in a list resolves directly against the corresponding measurement
    table. The check below confirms that.
    """)
    code(cells, """
    missing_301 = set(list_301.feature_name) - set(pd.read_csv(paths.WT_MORPHAGENT, nrows=1).columns)
    missing_400 = set(list_400.feature_name) - set(pd.read_csv(paths.PRED_MORPHAGENT, nrows=1).columns)
    print(f"301 features absent from the WT table         : {len(missing_301)}")
    print(f"400 features absent from the prediction table : {len(missing_400)}")
    assert not missing_301 and not missing_400
    print("\\nBoth feature lists resolve against the bundled tables.")
    """)

    # ------------------------------------------------------------ section 2
    md(cells, """
    ---
    ## 2. The Tau feature space MorphAgent produces

    **Claim under test.** MorphAgent features recover the organisation that
    expert-designed descriptors capture, while spanning a substantially richer
    phenotypic space.

    Both spaces are measured on the same paired WT Tau discovery cohort, each
    z-scored and projected onto its own first two principal components. Cells
    are coloured and sized by mean Tau intensity within the cell mask, on a log
    scale, so the abundance gradient is visible in both panels.
    """)
    code(cells, """
    cohort_wt = tau_pca.load_wt_cohort()

    print(f"cells               : {cohort_wt['morphagent'].shape[0]}")
    print(f"MorphAgent features : {cohort_wt['morphagent'].shape[1]}")
    print(f"expert features     : {cohort_wt['expert'].shape[1]}")
    print(f"Tau intensity range : {cohort_wt['intensity'].min():.0f} to {cohort_wt['intensity'].max():.0f}")
    print("\\nthe expert panel, in full:")
    for name in cohort_wt["expert"].columns:
        print(f"  {name}")
    """)
    code(cells, """
    import importlib
    importlib.reload(tau_pca)
    png_path, pca_summary = tau_pca.plot_feature_space_comparison(cohort_wt)
    display(Image(filename=str(png_path)))
    print(pca_summary.to_string(index=False))
    """)

    md(cells, """
    The grey ellipses are 95% covariance ellipses of the low- and
    high-intensity halves of the cohort (median split on Tau intensity),
    included as a visual guide to the abundance gradient rather than as a
    fitted result.

    The quantitative handle on this panel is how variance is distributed. A
    16-feature expert panel is close to one dominant axis of Tau abundance, so
    it concentrates far more variance in PC1. The MorphAgent space spreads
    variance across many more directions — that difference in effective
    dimensionality is the point of the comparison.
    """)
    code(cells, """
    checks["§2 feature space"] = tau_pca.compare_pca_to_published(pca_summary)
    print(checks["§2 feature space"].to_string(index=False))
    assert checks["§2 feature space"].filter(like="_match").all().all()
    print("\\nBoth projections match the published explained variance.")
    """)

    # ------------------------------------------------------------ section 3
    md(cells, """
    ---
    ## 3. WT versus the P301S+S320F double mutant

    **Claim under test.** MorphAgent features discriminate genotype better than
    the expert panel in both imaging modalities, and the advantage is largest
    under super-resolution.

    The cohort is an independent mutant dataset in which every cell was imaged
    twice — once at super-resolution (SR) and once wide-field (WF). Imaging
    modality and genotype are both encoded in the cell identifier, so labels are
    derived rather than supplied separately.
    """)
    code(cells, """
    tables = tc.load_cohort()
    print(tc.cohort_overview(tables).to_string(index=False))

    print("\\nhow labels are read off an identifier:")
    for sid in tables["morphagent"]["sample_id"].head(3):
        print(f"  {sid:<28} modality={tc.modality_from_id(sid):<3} genotype={tc.genotype_from_id(sid)}")
    """)

    md(cells, """
    All four panels share one classifier setup: median imputation, z-scoring,
    and balanced logistic regression scored by five-fold stratified
    cross-validation.

    The two feature families differ only in how many features enter. The expert
    panel uses all of its features. MorphAgent is first narrowed to the 50
    descriptors with the highest single-feature cross-validated AUC on the SR
    subset, and that same set of 50 is then applied unchanged to both
    modalities — so the wide-field result is not tuned on wide-field data.
    """)
    code(cells, """
    selection = tc.select_topk_on_sr(tables["morphagent"], top_k=tc.TOP_K_MORPHAGENT)
    print(f"{len(selection)} MorphAgent features selected. Top 15 by single-feature CV AUC:")
    print(selection.head(15).to_string(index=False))
    """)

    md(cells, """
    Each panel is a two-dimensional view of the fitted model: the horizontal
    axis is the signed distance to the separating hyperplane, so the dashed
    decision line sits at zero, and the vertical axis is the leading principal
    component of the subspace orthogonal to it.
    """)
    code(cells, """
    binary = tc.run_binary_panels(tables, top_k=tc.TOP_K_MORPHAGENT)
    png_path = tc.plot_binary_panels(binary)
    display(Image(filename=str(png_path)))
    print(binary["metrics"].to_string(index=False))
    """)
    code(cells, """
    checks["§3 binary"] = tc.compare(
        binary["metrics"].rename(
            columns={"cv_accuracy": "cv_accuracy_mean", "cv_roc_auc": "cv_roc_auc_mean"}
        ),
        tc.reference_binary_metrics(),
        ["cv_accuracy_mean", "cv_roc_auc_mean"],
    )
    print(checks["§3 binary"][[
        "feature_family", "modality",
        "cv_accuracy_mean_reproduced", "cv_accuracy_mean_published", "cv_accuracy_mean_match_3dp",
        "cv_roc_auc_mean_reproduced", "cv_roc_auc_mean_published", "cv_roc_auc_mean_match_3dp",
    ]].to_string(index=False))

    assert checks["§3 binary"].filter(like="match_3dp").all().all()
    print("\\nAll eight values match the manuscript to three decimal places.")
    """)

    md(cells, """
    MorphAgent leads in both modalities. The informative contrast is how much
    each family loses when it drops from super-resolution to wide-field: the
    expert panel falls by roughly 0.20 accuracy, MorphAgent by about 0.07. Much
    of the expert panel's discriminative signal therefore lives in nanoscale
    Tau organisation that wide-field blurs away, whereas MorphAgent keeps usable
    signal at both scales.
    """)

    # ------------------------------------------------------------ section 4
    md(cells, """
    ---
    ## 4. Six-class Tau genotype discrimination

    **Claim under test.** The expanded feature space improves genotype
    resolution beyond a two-class contrast.

    This task spans WT and five Tau mutants and uses every available feature in
    each family — no selection step. Alongside accuracy we report the macro
    one-vs-rest false-positive rate, averaged over classes, where lower is
    better.
    """)
    code(cells, """
    sixclass = tc.run_sixclass(tables)
    print(sixclass["summary"].to_string(index=False))
    """)

    md(cells, """
    Significance comes from a paired stratified bootstrap over the out-of-fold
    predictions. Every cell is predicted exactly once, in its held-out fold;
    cells are then resampled within genotype and both arms are scored on the
    same resampled cells, which is possible because each cell was imaged in both
    modalities. Comparisons are reported in the table below the figure rather
    than annotated on the bars.
    """)
    code(cells, """
    import importlib
    importlib.reload(tc)
    png_path, sixclass_significance = tc.plot_sixclass(sixclass, n_bootstrap=2000)
    display(Image(filename=str(png_path)))
    print(sixclass_significance.to_string(index=False))
    """)
    code(cells, """
    checks["§4 six-class"] = tc.compare(
        sixclass["summary"], tc.reference_sixclass_metrics(), ["accuracy", "macro_fpr"]
    )
    print(checks["§4 six-class"][[
        "feature_family", "modality",
        "accuracy_reproduced", "accuracy_published", "accuracy_match_3dp",
        "macro_fpr_reproduced", "macro_fpr_published", "macro_fpr_match_3dp",
    ]].to_string(index=False))

    assert checks["§4 six-class"].filter(like="match_3dp").all().all()
    print("\\nAll eight values match the manuscript to three decimal places.")
    """)

    md(cells, """
    MorphAgent gains about 0.17 accuracy over the expert panel at
    super-resolution while cutting the macro false-positive rate, and both
    gaps are significant. The super-resolution-versus-wide-field difference
    within MorphAgent is not significant under this test, which the table
    reports rather than omits.

    Summary metrics cannot say *which* genotypes get confused, so the pooled
    cross-validated confusion matrices are worth a look.
    """)
    code(cells, """
    for family in ("expert", "morphagent"):
        print(f"\\n{tc.FAMILY_LABEL[family]} — super-resolution (rows = true, columns = predicted)")
        print(tc.confusion_table(sixclass, family, "SR").to_string())
    """)

    # ------------------------------------------------------------ section 5
    md(cells, """
    ---
    ## 5. Predicting gene expression from Tau morphology

    **Claim under test.** Tau morphology carries transcriptomic information that
    Tau abundance alone does not.

    Morphology and transcriptome were measured on the same cells. Three
    predictor sets are compared gene by gene, with Tau intensity as the
    single-feature baseline that abundance-only explanations would predict is
    sufficient.

    | Arm | Predictors |
    |---|---|
    | MorphAgent | the 400 descriptors from `source/feature_lists` |
    | Expert | the handcrafted reference panel |
    | Tau intensity | the single mean-intensity readout |

    Set `MAX_GENES` to an integer for a fast pass; leave it at `None` to score
    the full transcriptome, which is what the published comparison requires.
    """)
    code(cells, """
    # None runs the full transcriptome (~7 min on 24 workers); an integer caps the gene count.
    MAX_GENES = None
    N_JOBS = 24

    cohort_pred = tp.load_cohort(max_genes=MAX_GENES)
    splits = tp.load_splits(10)

    print(f"cells : {len(cohort_pred['sample_ids'])}")
    print(f"genes : {len(cohort_pred['genes'])}")
    for arm in tp.ARMS:
        print(f"  {arm:<14} {cohort_pred['X'][arm].shape[1]:>3} predictors")
    sizes = {k: len(splits[0][k]) for k in ("train_idx", "val_idx", "test_idx")}
    print(f"\\nresamples    : {len(splits)}")
    print(f"per resample : {sizes}")
    """)

    md(cells, """
    ### 5.1 Evaluation protocol

    Each arm is evaluated with the procedure used for it in the manuscript, and
    those procedures are not identical — `tau_prediction.ARM_PROTOCOL` records
    which applies to which arm.

    **MorphAgent.** For each gene and resample, the 400 predictors are ranked by
    absolute Pearson correlation with the target using train and validation
    cells only. A support-vector regressor is fitted on the top *k* predictors
    for *k* in {1, 2, 4, 8, 16, 32, 64} and scored on the held-out test cells.
    The two best-scoring *k* are averaged into an ensemble, whose mean test
    correlation is the gene's score.

    **Expert and Tau intensity.** All predictors of the arm are used with no
    ranking step. Ten regressor configurations are drawn from a fixed grid and
    chosen by Spearman correlation on the inner validation cells, then refitted
    on train and validation and scored on test.
    """)
    code(cells, """
    for arm in tp.ARMS:
        print(f"  {arm:<14} {tp.ARM_PROTOCOL[arm]}")
    print(f"\\nk values searched by the MorphAgent arm : {tp.K_VALUES}")
    print(f"ensemble size                            : {tp.ENSEMBLE_TOP_N}")
    print(f"configuration grid size                  : {len(tp.svr_grid())}")
    print(f"configurations tried per gene/resample   : {tp.N_PARAM_TRIALS}")
    """)

    md(cells, """
    ### 5.2 Score every gene

    This is the expensive cell. Per-gene scores are written to `data/outputs/`
    so the cells below can be re-run without refitting anything.
    """)
    code(cells, """
    per_gene = tp.score_all_arms(cohort_pred, splits, n_jobs=N_JOBS)

    for arm, df in per_gene.items():
        df.to_csv(tp.OUTPUT_DIR / f"fig5a_per_gene_{arm.lower().replace(' ', '_')}.csv", index=False)
        print(f"{arm:<14} {len(df):>6} genes, mean r = {np.nanmean(df.mean_pearson):+.4f}")
    """)

    md(cells, """
    ### 5.3 Build the panel

    Each arm is ranked by its own scores, so the top-5,000 and top-1,000 panels
    use a different gene set per arm rather than a shared one. A gene whose
    correlation is undefined counts as zero, which keeps all three panels
    averaging over the same gene universe.
    """)
    code(cells, """
    summary_pred, raw_pred = tp.summarize(per_gene)
    summary_pred.to_csv(tp.OUTPUT_DIR / "fig5a_summary.csv", index=False)

    png_path, pred_significance = tp.plot_prediction_panel(summary_pred, raw_pred)
    display(Image(filename=str(png_path)))
    print(summary_pred.to_string(index=False))
    """)

    md(cells, """
    ### 5.4 Agreement with the published values

    The MorphAgent arm is deterministic and reproduces the published per-gene
    scores exactly. The expert and Tau-intensity arms draw ten regressor
    configurations per gene from a seeded generator, so individual gene scores
    shift slightly and the panel means land within about 0.002 of the published
    values rather than on them.
    """)
    code(cells, """
    check_pred = tp.compare_to_published(summary_pred)
    if MAX_GENES is not None:
        print("Skipped: set MAX_GENES = None to compare against the published all-gene values.")
    else:
        checks["§5 prediction"] = check_pred
        print(check_pred.to_string(index=False))
        assert check_pred.agrees.all()
        print("\\nEvery panel agrees with the manuscript within tolerance.")
    """)

    md(cells, """
    Two things stand out. Tau intensity sits at zero across the transcriptome
    while the morphology arms are positive, so abundance alone carries
    essentially no transcriptomic signal — the information is in the spatial
    organisation. And the signal is strongly concentrated: restricting to each
    arm's best-predicted genes raises mean correlation several-fold, so most of
    the transcriptome is not predictable from morphology while a well-defined
    subset is, with MorphAgent predicting that subset markedly better.

    Per-gene scores are in `data/outputs/fig5a_per_gene_*.csv` if you want to
    inspect individual genes or which *k* each one selected.
    """)
    code(cells, """
    morph_scores = per_gene["MorphAgent"]
    print("Best-predicted genes in the MorphAgent arm:")
    print(morph_scores.head(15)[["own_rank", "gene", "mean_pearson", "best_k", "ensemble_ks"]]
          .to_string(index=False))

    print("\\nHow often each k was selected:")
    print(morph_scores.best_k.value_counts().sort_index().to_string())
    """)

    # ------------------------------------------------------------ section 6
    md(cells, """
    ---
    ## 6. Top-level morphology–GO correspondence

    **Claim under test.** Image-defined Tau morphology modules and independently
    defined GO gene programs converge on the same genes.

    MorphAgent features were grouped into 11 morphology-defined biological
    processes. Each class is assigned the union of genes correlated with its
    member features (`|ρ| ≥ 0.4`). Each enriched GO term contributes its member
    genes; nine mitochondrial terms are merged into one column, and two redundant
    terms are dropped, giving an 11 × 11 matrix. The cell value is the
    **shared-gene count** — how many genes are jointly implicated by a
    morphology class and a GO program.
    """)
    code(cells, """
    morph_classes = th.load_morphology_classes()
    go_terms = th.load_go_terms()

    print(f"morphology classes : {len(morph_classes)}")
    print(f"GO terms (display) : {len(go_terms)}")
    print()
    print(morph_classes[["bp_id", "bp_display", "n_features", "n_genes"]].to_string(index=False))
    print()
    print(go_terms[["go_display_order", "go_short", "n_genes"]].to_string(index=False))
    """)
    code(cells, """
    top = th.top_level_overlap(morph_classes, go_terms)
    top.to_csv(paths.OUTPUT_DIR / "fig5d_bp_go_overlap_long.csv", index=False)

    print("boxed correspondences in the published panel:")
    print(th.highlight_summary(top).to_string(index=False))

    png_path = th.plot_top_level_heatmap(top)
    """)
    code(cells, """
    checks["§6 top-level heatmap"] = th.compare_top_level(top)
    n_ok = int(checks["§6 top-level heatmap"]["match"].sum())
    n_tot = len(checks["§6 top-level heatmap"])
    print(f"{n_ok} of {n_tot} cells match the published shared-gene counts.")
    assert n_ok == n_tot
    """)

    md(cells, """
    The RNA-centric block (mRNA processing, ribonucleoprotein complex
    biogenesis, spliceosomal complex, ribosome biogenesis) is the densest
    overlap. Two correspondences are boxed for the next section: Tau
    mislocalization × mRNA processing / spliceosomal complex, and
    Tau-positive object maturation × ribonucleoprotein complex biogenesis /
    lysosomal membrane.
    """)

    # ------------------------------------------------------------ section 7
    md(cells, """
    ---
    ## 7. Middle-level architecture–GRN correspondence

    **Claim under test.** The boxed top-level pairs resolve into specific
    subcellular architectures and GRN regulatory modules.

    Each morphology class is split by structure-size into subcellular
    architectures; each architecture's gene set is the union of genes
    correlated with its features. On the gene side, GRN modules are the
    transcription factor plus its top-decile targets inside the GO term. The
    two panels expand:

    | Morphology class | GO term | GRNs |
    |---|---|---|
    | Tau-positive object maturation (BP06) | ribonucleoprotein complex biogenesis | 8 |
    | Tau mislocalization (BP01) | spliceosomal complex | 3 |
    """)
    code(cells, """
    feat2genes = th.load_feature_genes()
    print(f"features with correlated genes: {len(feat2genes)}")

    for job in th.MIDDLE_JOBS:
        sas = th.architecture_gene_sets(job["bp_id"], feat2genes)
        grns = th.load_grns(job["go_term"])
        print(f"\\n{job['bp_display']}  ({job['bp_id']}, {len(sas)} architectures)")
        print(pd.DataFrame(sas)[["sa_local_id", "architecture", "n_features", "n_genes"]].to_string(index=False))
        print(f"\\nGRNs in {job['go_term']}:")
        print(grns[["GRN_id", "TF", "targets", "n_genes"]].to_string(index=False))
    """)
    code(cells, """
    middle = {}
    for job in th.MIDDLE_JOBS:
        df = th.middle_level_overlap(job, feat2genes)
        df.to_csv(paths.OUTPUT_DIR / f"fig5e_{job['bp_id'].lower()}_overlap_long.csv", index=False)
        middle[job["bp_id"]] = df
        nz = df[df["shared_gene_count"] > 0]
        print(f"\\n{job['bp_display']} × {job['go_term']}  ({len(nz)} non-zero cells)")
        print(nz[[
            "architecture_display", "GRN_id", "GRN_TF", "shared_gene_count", "shared_genes"
        ]].to_string(index=False))

    png_path = th.plot_middle_level_panel(middle)
    """)

    md(cells, """
    On the left, overlap is size-graded: small puncta share RPS15, cytoplasmic
    architectures pick up RPP25 / EIF3K / TSR2, and the boundary architecture
    shares two genes (EIF3K, SNRPC) with the SNRPB2 module. On the right, Tau
    mislocalization maps onto the spliceosomal circuit at the perinuclear ring
    (SNRPC, RNPC3), the nuclear boundary (RALY), and whole-cell geometry (SUGP1).
    """)

    # ---------------------------------------------------------------- close
    md(cells, """
    ---
    ## 8. All reproduced numbers in one place
    """)
    code(cells, """
    lines = []
    for label, frame in checks.items():
        cols = [c for c in frame.columns if c.endswith(("_match", "_match_3dp", "agrees"))]
        total = int(frame[cols].size)
        passed = int(frame[cols].to_numpy().sum())
        lines.append({"section": label, "values_checked": total, "agreeing": passed})

    audit = pd.DataFrame(lines)
    print(audit.to_string(index=False))
    print(f"\\n{audit.agreeing.sum()} of {audit.values_checked.sum()} reported values reproduced.")
    print(f"\\nFigures and tables written to: {tp.rel(tp.OUTPUT_DIR)}")
    """)
    md(cells, """
    Taken together: MorphAgent's designed features span a much wider Tau
    phenotypic space than a curated expert panel, convert that width into better
    genotype discrimination in both a two-class and a six-class setting, recover
    transcriptomic structure that Tau abundance alone misses, and organise that
    shared component as a matched morphology–gene hierarchy — biological
    processes versus GO functions at the top, subcellular architectures versus
    GRN modules in the middle.
    """)

    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    path = ROOT / NOTEBOOK_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(path))
    print(f"wrote {path} ({len(cells)} cells)")
    return path


if __name__ == "__main__":
    build()
