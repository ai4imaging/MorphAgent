#!/usr/bin/env python3
"""Figure 5a — gene expression prediction from Tau morphology.

Three predictor sets are compared on the paired morphology/transcriptome
cohort, gene by gene:

  * MorphAgent  : the 400 descriptors in source/feature_lists (200 code-derived
                  plus 200 vision-language descriptors)
  * Expert      : the handcrafted reference panel
  * Tau intensity : the single mean-intensity readout

Protocol
--------
Ten fixed train / validation / test resamples ship in source/splits. Each arm
is evaluated with the procedure used for it in the manuscript.

MorphAgent arm, per gene and per resample:
  1. rank the 400 predictors by |Pearson| against the target, using
     train+validation cells only;
  2. fit an RBF support-vector regressor on the top-k predictors for
     k in {1, 2, 4, 8, 16, 32, 64};
  3. score every k by Pearson correlation on the held-out test cells.
Across resamples the two best-scoring k are averaged into an ensemble, and the
gene's score is that ensemble's mean test Pearson.

Expert and Tau-intensity arms, per gene and per resample: all predictors of the
arm are used with no ranking step; ten support-vector-regressor configurations
are drawn from a fixed grid and chosen by Spearman correlation on the inner
validation cells, then refitted on train+validation and scored on test.

Bars report the mean across genes, each arm ranked by its own scores, for all
genes and for that arm's own top 5,000 and top 1,000 genes.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu, pearsonr, sem, spearmanr
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from paths import (
    CACHED_RESULTS,
    LIST_400,
    OUTPUT_DIR,
    PRED_EXPERT,
    PRED_MORPHAGENT,
    PRED_TRANSCRIPTOME,
    SPLIT_DIR,
    TAU_INTENSITY_FEATURE,
    rel,
    require,
)

def _silence_expected_warnings() -> None:
    """Constant targets and unconverged SVR fits are expected for some genes."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    for name in ("ConstantInputWarning", "NearConstantInputWarning"):
        category = getattr(__import__("scipy.stats", fromlist=[name]), name, None)
        if category is not None:
            warnings.filterwarnings("ignore", category=category)


_silence_expected_warnings()

K_VALUES = [1, 2, 4, 8, 16, 32, 64]
ENSEMBLE_TOP_N = 2
N_PARAM_TRIALS = 10
PANELS = ["All genes", "Top-5000 genes", "Top-1000 genes"]
ARMS = ["MorphAgent", "Expert", "Tau intensity"]
ARM_COLORS = {"MorphAgent": "#2f6f9f", "Expert": "#8e6bb5", "Tau intensity": "#d4896f"}

# Which procedure each arm was evaluated with in the manuscript.
ARM_PROTOCOL = {
    "MorphAgent": "screen_topk_ensemble",
    "Expert": "all_features_param_search",
    "Tau intensity": "all_features_param_search",
}
# Tag strings enter the per-gene random seed, so runs are reproducible.
ARM_SEED_TAG = {"Expert": "expert_features", "Tau intensity": "tau_expression"}


def configure_mpl() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "sans-serif"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _id_column(df: pd.DataFrame) -> str:
    for col in ("sample_id", "cell_id", "CellID", "cellID"):
        if col in df.columns:
            return col
    return str(df.columns[0])


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.apply(pd.to_numeric, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.fillna(out.median()).fillna(0.0).clip(-1e9, 1e9)


def load_feature_list_400() -> pd.DataFrame:
    return pd.read_csv(require(LIST_400))


def load_splits(n_repeats: int = 10) -> List[Dict[str, np.ndarray]]:
    paths = sorted(SPLIT_DIR.glob("repeat_*.json"))[:n_repeats]
    if not paths:
        raise FileNotFoundError(f"No repeat_*.json under {SPLIT_DIR}")
    splits = []
    for p in paths:
        sp = json.loads(p.read_text())
        splits.append(
            {
                "repeat_id": int(sp["repeat_id"]),
                "train_idx": np.asarray(sp["train_idx"], dtype=int),
                "val_idx": np.asarray(sp["val_idx"], dtype=int),
                "test_idx": np.asarray(sp["test_idx"], dtype=int),
                "inner_train_idx": np.asarray(sp["inner_train_idx"], dtype=int),
                "inner_val_idx": np.asarray(sp["inner_val_idx"], dtype=int),
            }
        )
    return splits


def load_cohort(max_genes: Optional[int] = None) -> Dict[str, object]:
    """Align the three predictor tables and the transcriptome on sample id."""
    morph = pd.read_csv(require(PRED_MORPHAGENT))
    expert = pd.read_csv(require(PRED_EXPERT))
    omics = pd.read_csv(require(PRED_TRANSCRIPTOME))

    frames = {}
    for name, df in (("morphagent", morph), ("expert", expert), ("omics", omics)):
        df = df.rename(columns={_id_column(df): "sample_id"})
        df["sample_id"] = df["sample_id"].astype(str)
        frames[name] = df

    common = sorted(
        set(frames["morphagent"]["sample_id"])
        & set(frames["expert"]["sample_id"])
        & set(frames["omics"]["sample_id"])
    )
    for name in frames:
        frames[name] = (
            frames[name][frames[name]["sample_id"].isin(common)]
            .sort_values("sample_id")
            .reset_index(drop=True)
        )

    names_400 = load_feature_list_400()["feature_name"].astype(str).tolist()
    morph_cols = [c for c in names_400 if c in frames["morphagent"].columns]
    if len(morph_cols) != len(names_400):
        missing = sorted(set(names_400) - set(morph_cols))
        raise KeyError(f"{len(missing)} of the 400 features are absent, e.g. {missing[:5]}")

    expert_cols = [c for c in frames["expert"].columns if c != "sample_id"]
    if TAU_INTENSITY_FEATURE not in expert_cols:
        raise KeyError(f"Expected '{TAU_INTENSITY_FEATURE}' in the expert table")

    genes = [c for c in frames["omics"].columns if c != "sample_id"]
    if max_genes is not None:
        genes = genes[:max_genes]

    return {
        "sample_ids": common,
        "genes": genes,
        "X": {
            "MorphAgent": clean_numeric(frames["morphagent"][morph_cols]).to_numpy(float),
            "Expert": clean_numeric(frames["expert"][expert_cols]).to_numpy(float),
            "Tau intensity": clean_numeric(
                frames["expert"][[TAU_INTENSITY_FEATURE]]
            ).to_numpy(float),
        },
        "feature_names": {
            "MorphAgent": morph_cols,
            "Expert": expert_cols,
            "Tau intensity": [TAU_INTENSITY_FEATURE],
        },
        "Y": clean_numeric(frames["omics"][genes]).to_numpy(float),
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Undefined correlations are left undefined; the caller averages with nanmean."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    r = pearsonr(a, b)[0]
    return float(r) if np.isfinite(r) else float("nan")


def _pearson_or_zero(a: np.ndarray, b: np.ndarray) -> float:
    """Undefined correlations count as zero, which shrinks that gene toward zero.

    This is the convention the expert and Tau-intensity arms were scored with:
    a resample whose prediction is constant contributes 0 rather than dropping
    out of the gene's average.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0 or np.nanstd(a) <= 1e-12 or np.nanstd(b) <= 1e-12:
        return 0.0
    r = pearsonr(a, b)[0]
    return float(r) if np.isfinite(r) else 0.0


def _make_svr() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svr", SVR(C=1.0, gamma="scale", kernel="rbf", epsilon=0.1, max_iter=5000)),
        ]
    )


def _screen_order(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Rank predictors by |Pearson| with the target, using train cells only."""
    y = np.asarray(y_train, float) - np.mean(y_train)
    y_std = y.std()
    if y_std == 0 or not np.isfinite(y_std):
        return np.arange(x_train.shape[1])
    x = np.asarray(x_train, float)
    x = x - x.mean(axis=0)
    x_std = x.std(axis=0).astype(float)
    x_std[~np.isfinite(x_std) | (x_std == 0)] = np.nan
    corr = (x.T @ y) / (x_std * y_std * len(y))
    abs_corr = np.abs(corr)
    abs_corr[~np.isfinite(abs_corr)] = -1.0
    return np.argsort(-abs_corr)


def _score_one_gene(
    gene_idx: int,
    gene: str,
    X: np.ndarray,
    Y: np.ndarray,
    splits: Sequence[Dict[str, np.ndarray]],
    k_values: Sequence[int],
) -> Dict[str, object]:
    _silence_expected_warnings()
    y = Y[:, gene_idx]
    per_k: Dict[int, List[float]] = {k: [] for k in k_values}
    cached: List[Tuple[np.ndarray, Dict[int, np.ndarray]]] = []

    for sp in splits:
        train = np.concatenate([sp["train_idx"], sp["val_idx"]])
        test = sp["test_idx"]
        order = _screen_order(X[train], y[train])
        preds: Dict[int, np.ndarray] = {}
        for k in k_values:
            feats = order[:k]
            try:
                model = _make_svr()
                model.fit(X[train][:, feats], y[train])
                pred = model.predict(X[test][:, feats])
                per_k[k].append(_pearson(y[test], pred))
                preds[k] = np.asarray(pred, float)
            except Exception:
                per_k[k].append(float("nan"))
                preds[k] = np.full(len(test), np.nan)
        cached.append((test, preds))

    mean_k = {k: float(np.nanmean(per_k[k])) for k in k_values}
    best_ks = sorted(
        k_values, key=lambda k: (-(mean_k[k] if np.isfinite(mean_k[k]) else -1e18), k)
    )[:ENSEMBLE_TOP_N]

    ens = [
        _pearson(y[test], np.nanmean(np.vstack([preds[k] for k in best_ks]), axis=0))
        for test, preds in cached
    ]
    row = {
        "gene": gene,
        "best_k": int(best_ks[0]),
        "ensemble_ks": "|".join(str(k) for k in best_ks),
        "mean_pearson": float(np.nanmean(ens)),
    }
    row.update({f"mean_pearson_k{k}": mean_k[k] for k in k_values})
    return row


def svr_grid() -> List[Dict[str, Any]]:
    """The fixed configuration grid searched by the expert and Tau arms."""
    return [
        {"C": c, "gamma": g, "kernel": k, "epsilon": e, "pca_nc": p}
        for c, g, k, e, p in itertools.product(
            [0.1, 1.0, 10.0, 100.0],
            ["scale", "auto", 0.01, 0.1],
            ["rbf", "linear"],
            [0.01, 0.1],
            [None, 0.95],
        )
    ]


def _build_model(params: Dict[str, Any]) -> Pipeline:
    steps: List[Tuple[str, Any]] = [("scaler", StandardScaler())]
    if params.get("pca_nc") is not None:
        steps.append(("pca", PCA(n_components=params["pca_nc"])))
    steps.append(
        (
            "svr",
            SVR(
                C=float(params["C"]),
                gamma=params["gamma"],
                kernel=str(params["kernel"]),
                epsilon=float(params["epsilon"]),
                max_iter=10000,
            ),
        )
    )
    return Pipeline(steps)


def _trials_for(gene_idx: int, repeat_id: int, tag: str, n_trials: int) -> List[Dict[str, Any]]:
    grid = svr_grid()
    rng = np.random.default_rng(
        42 + 10_007 + int(repeat_id) * 1_003 + gene_idx * 97 + sum(ord(c) for c in tag)
    )
    picks = rng.choice(len(grid), size=min(n_trials, len(grid)), replace=False)
    return [grid[int(i)] for i in picks]


def _score_one_gene_param_search(
    gene_idx: int,
    gene: str,
    X: np.ndarray,
    Y: np.ndarray,
    splits: Sequence[Dict[str, np.ndarray]],
    tag: str,
    n_trials: int = N_PARAM_TRIALS,
) -> Dict[str, object]:
    """Expert / Tau arm: no predictor ranking, configuration picked on inner validation."""
    _silence_expected_warnings()
    y = Y[:, gene_idx]
    scores, chosen = [], []
    for sp in splits:
        best, best_score = None, -np.inf
        itr, iva = sp["inner_train_idx"], sp["inner_val_idx"]
        for params in _trials_for(gene_idx, sp["repeat_id"], tag, n_trials):
            try:
                model = _build_model(params)
                model.fit(X[itr], y[itr])
                score = spearmanr(y[iva], model.predict(X[iva]))[0]
                score = 0.0 if not np.isfinite(score) else float(score)
            except Exception:
                continue
            if score > best_score:
                best, best_score = params, score
        if best is None:
            best = {"C": 1.0, "gamma": "scale", "kernel": "rbf", "epsilon": 0.1, "pca_nc": None}
        chosen.append(f"{best['kernel']}|C={best['C']}")
        train = np.concatenate([sp["train_idx"], sp["val_idx"]])
        try:
            model = _build_model(best)
            model.fit(X[train], y[train])
            scores.append(_pearson_or_zero(y[sp["test_idx"]], model.predict(X[sp["test_idx"]])))
        except Exception:
            scores.append(0.0)
    return {
        "gene": gene,
        "mean_pearson": float(np.mean(scores)) if scores else 0.0,
        "modal_config": max(set(chosen), key=chosen.count) if chosen else "",
    }


def score_arm(
    arm: str,
    cohort: Dict[str, object],
    splits: Sequence[Dict[str, np.ndarray]],
    n_jobs: int = 8,
    verbose: bool = True,
) -> pd.DataFrame:
    X = cohort["X"][arm]
    Y = cohort["Y"]
    genes = cohort["genes"]
    protocol = ARM_PROTOCOL[arm]
    t0 = time.time()

    if protocol == "screen_topk_ensemble":
        k_values = [k for k in K_VALUES if k <= X.shape[1]] or [X.shape[1]]
        if verbose:
            print(
                f"[{arm}] genes={len(genes)} predictors={X.shape[1]} "
                f"k={k_values} repeats={len(splits)} n_jobs={n_jobs}",
                flush=True,
            )
        rows = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_score_one_gene)(i, g, X, Y, splits, k_values) for i, g in enumerate(genes)
        )
    else:
        tag = ARM_SEED_TAG[arm]
        if verbose:
            print(
                f"[{arm}] genes={len(genes)} predictors={X.shape[1]} (no ranking) "
                f"config trials={N_PARAM_TRIALS} repeats={len(splits)} n_jobs={n_jobs}",
                flush=True,
            )
        rows = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_score_one_gene_param_search)(i, g, X, Y, splits, tag)
            for i, g in enumerate(genes)
        )

    df = pd.DataFrame(rows).sort_values(["mean_pearson", "gene"], ascending=[False, True])
    df = df.reset_index(drop=True)
    df.insert(0, "own_rank", np.arange(1, len(df) + 1))
    df.insert(1, "feature_set", arm)
    df.insert(2, "protocol", protocol)
    if verbose:
        print(
            f"[{arm}] done in {time.time() - t0:.1f}s, "
            f"mean={np.nanmean(df.mean_pearson):.4f}",
            flush=True,
        )
    return df


def score_all_arms(
    cohort: Dict[str, object],
    splits: Sequence[Dict[str, np.ndarray]],
    n_jobs: int = 8,
) -> Dict[str, pd.DataFrame]:
    return {arm: score_arm(arm, cohort, splits, n_jobs=n_jobs) for arm in ARMS}


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

def _panel_values(scores: pd.Series, top: Optional[int]) -> np.ndarray:
    """A gene whose score is undefined counts as zero rather than disappearing,
    so every panel is an average over the same gene universe. Such genes sit at
    the bottom of the ranking and never enter the top-k panels."""
    s = scores.fillna(0.0).sort_values(ascending=False)
    return (s if top is None else s.head(top)).to_numpy(float)


def summarize(per_gene: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict]:
    tops = [None, 5000, 1000]
    rows, raw = [], {}
    for panel, top in zip(PANELS, tops):
        raw[panel] = {}
        for arm in ARMS:
            vals = _panel_values(per_gene[arm].set_index("gene")["mean_pearson"], top)
            raw[panel][arm] = vals
            rows.append(
                {
                    "panel": panel,
                    "feature_set": arm,
                    "n_genes": int(len(vals)),
                    "mean_pearson": float(np.nanmean(vals)),
                    "sem": float(sem(vals)) if len(vals) > 1 else 0.0,
                    "median_pearson": float(np.nanmedian(vals)),
                }
            )
    return pd.DataFrame(rows), raw


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def plot_prediction_panel(
    summary: pd.DataFrame, raw: Dict, out_stem: Optional[Path] = None
) -> Tuple[Path, pd.DataFrame]:
    configure_mpl()
    out_stem = out_stem or (OUTPUT_DIR / "fig5a_gene_expression_prediction")
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.9), sharey=True)
    fig.suptitle(
        "Gene expression prediction from MorphAgent morphology features",
        fontsize=13, y=1.04,
    )
    sig_rows = []
    ceiling = 0.0
    for ax, panel in zip(axes, PANELS):
        block = summary[summary.panel == panel].set_index("feature_set").loc[ARMS]
        vals = block["mean_pearson"].to_numpy(float)
        errs = block["sem"].to_numpy(float)
        x = np.arange(len(ARMS))
        ax.bar(x, vals, yerr=errs, capsize=3,
               color=[ARM_COLORS[a] for a in ARMS], width=0.72, zorder=2)
        for xi, v, e in zip(x, vals, errs):
            ax.text(xi, v + e + 0.014 if v >= 0 else v - e - 0.022, f"{v:.3f}",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=9)

        base = float(np.nanmax(vals + errs)) + 0.05
        for level, other in enumerate(("Expert", "Tau intensity")):
            p = float(
                mannwhitneyu(raw[panel]["MorphAgent"], raw[panel][other],
                             alternative="two-sided").pvalue
            )
            stars = _stars(p)
            y = base + level * 0.055
            b = ARMS.index(other)
            ax.plot([0, 0, b, b], [y, y + 0.013, y + 0.013, y], color="#222222", lw=0.9)
            ax.text(b / 2, y + 0.017, stars, ha="center", va="bottom", fontsize=10)
            ceiling = max(ceiling, y + 0.07)
            sig_rows.append(
                {"panel": panel, "comparison": f"MorphAgent vs {other}",
                 "p_mannwhitney": p, "stars": stars}
            )

        ax.set_title(panel.replace("Top-5000", "Top 5,000").replace("Top-1000", "Top 1,000"),
                     fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(["MorphAgent", "Expert", "Tau intensity"], rotation=20, ha="right",
                           fontsize=10)
        ax.grid(axis="y", color="#e8e8e8", lw=0.7)
        ax.set_axisbelow(True)
        ax.axhline(0, color="#bbbbbb", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Mean Pearson correlation across genes", fontsize=11)
    for ax in axes:
        ax.set_ylim(-0.05, max(0.78, ceiling))
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {rel(Path(str(out_stem)))}.png/.svg/.pdf")
    return Path(f"{out_stem}.png"), pd.DataFrame(sig_rows)


def reference_summary() -> Optional[pd.DataFrame]:
    path = CACHED_RESULTS / "gene_prediction_summary.csv"
    if not path.is_file():
        return None
    ref = pd.read_csv(path).rename(columns={"mean": "mean_pearson_published", "n": "n_genes"})
    return ref[["panel", "feature_set", "n_genes", "mean_pearson_published"]]


#: The MorphAgent arm is deterministic and reproduces the published per-gene
#: scores exactly. The expert and Tau arms draw their regressor configurations
#: per gene, so they land near rather than on the published value.
TOLERANCE = {"MorphAgent": 1e-6, "Expert": 2e-3, "Tau intensity": 2e-3}


def compare_to_published(summary: pd.DataFrame) -> Optional[pd.DataFrame]:
    ref = reference_summary()
    if ref is None:
        return None
    merged = summary.merge(ref, on=["panel", "feature_set"], suffixes=("", "_ref"))
    merged["abs_diff"] = (merged["mean_pearson"] - merged["mean_pearson_published"]).abs()
    merged["tolerance"] = merged["feature_set"].map(TOLERANCE)
    merged["agrees"] = merged["abs_diff"] <= merged["tolerance"]
    return merged[
        ["panel", "feature_set", "n_genes", "mean_pearson",
         "mean_pearson_published", "abs_diff", "tolerance", "agrees"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-genes", type=int, default=None,
                        help="Cap the gene count for a fast pass (default: all genes)")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort(max_genes=args.max_genes)
    splits = load_splits(args.repeats)
    print(f"Cohort: {len(cohort['sample_ids'])} cells, {len(cohort['genes'])} genes")
    for arm in ARMS:
        print(f"  {arm:<14} {cohort['X'][arm].shape[1]} predictors")

    per_gene = score_all_arms(cohort, splits, n_jobs=args.n_jobs)
    for arm, df in per_gene.items():
        tag = arm.lower().replace(" ", "_")
        df.to_csv(OUTPUT_DIR / f"fig5a_per_gene_{tag}.csv", index=False)

    summary, raw = summarize(per_gene)
    summary.to_csv(OUTPUT_DIR / "fig5a_summary.csv", index=False)
    _, sig = plot_prediction_panel(summary, raw)
    sig.to_csv(OUTPUT_DIR / "fig5a_significance.csv", index=False)

    print("\n" + summary.to_string(index=False))
    cmp_df = compare_to_published(summary)
    if cmp_df is not None and args.max_genes is None:
        print("\nAgainst the published values:")
        print(cmp_df.to_string(index=False))


if __name__ == "__main__":
    main()
