#!/usr/bin/env python3
"""Figure 4d / 4i — Tau genotype classification from morphology.

Cohort: an independent mutant dataset imaged twice per cell, once with
super-resolution (SR) and once wide-field (WF).

Two tasks, both 5-fold stratified cross-validation with balanced logistic
regression on median-imputed, z-scored features:

  4d  Binary WT versus P301S+S320F.
      MorphAgent uses the Top-50 descriptors ranked by single-feature CV AUC
      on the SR subset; the expert panel uses all of its features.
      Panels show the decision-function axis against the leading in-plane
      component, with the separating hyperplane at 0.

  4i  Six-class genotype discrimination (WT plus five Tau mutants), no feature
      selection, reported as accuracy and macro one-vs-rest false-positive rate.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from paths import CACHED_RESULTS, MUTANT_EXPERT, MUTANT_MORPHAGENT, OUTPUT_DIR, rel, require

RANDOM_STATE = 42
CV_SPLITS = 5
TOP_K_MORPHAGENT = 50

BINARY_MUTANT = "P301SandS320F"
BINARY_MUTANT_LABEL = "P301S+S320F"

SIX_CLASSES = ["WT", "S320F", "S305I", "P301LS320F", "P301SandS320F", "P301L"]
SIX_CLASS_LABELS = ["WT", "S320F", "S305I", "P301L+S320F", "P301S+S320F", "P301L"]

FAMILY_LABEL = {"expert": "expert-designed features", "morphagent": "MorphAgent"}
MODALITY_LABEL = {"SR": "Super-resolution", "WF": "Wide-field"}

COLOR_WT = "#1f77b4"
COLOR_MUT = "#d62728"
BAR_COLORS = {
    ("expert", "SR"): "#4E79A7",
    ("expert", "WF"): "#A6CEE3",
    ("morphagent", "SR"): "#D95F02",
    ("morphagent", "WF"): "#F4A582",
}


def configure_mpl() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "sans-serif"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42


# ---------------------------------------------------------------------------
# Cohort bookkeeping: modality and genotype are encoded in the sample id
# ---------------------------------------------------------------------------

def modality_from_id(sample_id: str) -> Optional[str]:
    s = str(sample_id).lower()
    if s.startswith("sr_"):
        return "SR"
    if s.startswith("wf_"):
        return "WF"
    return None


def genotype_from_id(sample_id: str) -> str:
    u = str(sample_id).upper()
    if re.match(r"^(SR|WF)_WT_", u):
        return "WT"
    if "P301S" in u and "S320F" in u:
        return "P301SandS320F"
    if "P301L" in u and "S320F" in u:
        return "P301LS320F"
    if "S305I" in u:
        return "S305I"
    if "MAPT" in u:
        return "MAPT"
    if "P301L" in u:
        return "P301L"
    if "S320F" in u:
        return "S320F"
    return "OTHER"


def load_cohort() -> Dict[str, pd.DataFrame]:
    morph = pd.read_csv(require(MUTANT_MORPHAGENT))
    expert = pd.read_csv(require(MUTANT_EXPERT))
    for df in (morph, expert):
        df["sample_id"] = df["sample_id"].astype(str)
    return {"morphagent": morph, "expert": expert}


def feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c != "sample_id"]


def cohort_overview(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for family, df in tables.items():
        ids = df["sample_id"].astype(str)
        for modality in ("SR", "WF"):
            mask = ids.map(lambda s: modality_from_id(s) == modality)
            counts = ids[mask].map(genotype_from_id).value_counts()
            rows.append(
                {
                    "feature_family": FAMILY_LABEL[family],
                    "modality": modality,
                    "n_features": len(feature_columns(df)),
                    "n_cells": int(mask.sum()),
                    **{c: int(counts.get(c, 0)) for c in SIX_CLASSES},
                }
            )
    return pd.DataFrame(rows)


def _design_matrix(sub: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    raw = sub[list(cols)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    return pipe.fit_transform(raw)


def _logreg(multiclass: bool = False) -> LogisticRegression:
    return LogisticRegression(
        solver="lbfgs" if multiclass else "liblinear",
        max_iter=4000 if multiclass else 8000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


# ---------------------------------------------------------------------------
# Figure 4d — binary WT vs P301S+S320F
# ---------------------------------------------------------------------------

def binary_subset(
    df: pd.DataFrame, modality: str, mutant: str = BINARY_MUTANT
) -> Tuple[pd.DataFrame, np.ndarray]:
    ids = df["sample_id"].astype(str)
    mask = ids.map(
        lambda s: modality_from_id(s) == modality and genotype_from_id(s) in {"WT", mutant}
    )
    sub = df.loc[mask].reset_index(drop=True)
    y = (
        sub["sample_id"]
        .astype(str)
        .map(lambda s: 0 if genotype_from_id(s) == "WT" else 1)
        .to_numpy(dtype=int)
    )
    return sub, y


def _single_feature_cv_auc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    keep = np.isfinite(x)
    x, yy = x[keep], y[keep].astype(int)
    if len(x) < 20 or len(np.unique(yy)) < 2:
        return float("nan")
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for tr, te in skf.split(x.reshape(-1, 1), yy):
        clf = LogisticRegression(
            solver="liblinear", max_iter=4000, class_weight="balanced", random_state=RANDOM_STATE
        )
        clf.fit(x[tr].reshape(-1, 1), yy[tr])
        prob = clf.predict_proba(x[te].reshape(-1, 1))[:, 1]
        auc = roc_auc_score(yy[te], prob)
        aucs.append(max(auc, 1 - auc))
    return float(np.mean(aucs))


def select_topk_on_sr(
    morph: pd.DataFrame, mutant: str = BINARY_MUTANT, top_k: int = TOP_K_MORPHAGENT
) -> pd.DataFrame:
    """Rank MorphAgent descriptors by single-feature CV AUC on the SR subset."""
    sub, y = binary_subset(morph, "SR", mutant)
    rows = []
    for col in feature_columns(morph):
        auc = _single_feature_cv_auc(pd.to_numeric(sub[col], errors="coerce").to_numpy(), y)
        if np.isfinite(auc):
            rows.append({"feature": col, "cv_auc_sr": auc})
    out = (
        pd.DataFrame(rows)
        .sort_values("cv_auc_sr", ascending=False)
        .reset_index(drop=True)
        .head(top_k)
    )
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def cv_binary_metrics(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    accs, aucs = [], []
    for tr, te in skf.split(X, y):
        clf = _logreg()
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        accs.append(accuracy_score(y[te], (prob >= 0.5).astype(int)))
        aucs.append(roc_auc_score(y[te], prob))
    return float(np.mean(accs)), float(np.mean(aucs))


def decision_projection(X: np.ndarray, w: np.ndarray, b: float) -> Tuple[np.ndarray, np.ndarray]:
    """Component 1 = signed distance to the hyperplane; Component 2 = leading
    principal component of the subspace orthogonal to the normal vector."""
    z1 = X @ w + b
    norm = np.linalg.norm(w)
    if norm < 1e-12 or X.shape[1] == 1:
        return z1, np.zeros(X.shape[0])
    u = w / norm
    X_orth = X - np.outer(X @ u, u)
    if np.allclose(X_orth, 0):
        return z1, np.zeros(X.shape[0])
    z2 = PCA(n_components=1, random_state=RANDOM_STATE).fit_transform(X_orth).ravel()
    return z1, z2


def run_binary_panels(
    tables: Dict[str, pd.DataFrame], top_k: int = TOP_K_MORPHAGENT
) -> Dict[str, object]:
    selection = select_topk_on_sr(tables["morphagent"], top_k=top_k)
    morph_cols = selection["feature"].tolist()
    expert_cols = feature_columns(tables["expert"])

    panels: Dict[Tuple[str, str], Dict] = {}
    rows = []
    for family, cols in (("expert", expert_cols), ("morphagent", morph_cols)):
        for modality in ("SR", "WF"):
            sub, y = binary_subset(tables[family], modality)
            X = _design_matrix(sub, cols)
            acc, auc = cv_binary_metrics(X, y)
            clf = _logreg()
            clf.fit(X, y)
            z1, z2 = decision_projection(X, clf.coef_.ravel(), float(clf.intercept_[0]))
            panels[(family, modality)] = {"z1": z1, "z2": z2, "y": y, "acc": acc, "auc": auc}
            rows.append(
                {
                    "feature_family": FAMILY_LABEL[family],
                    "modality": modality,
                    "n_features_used": len(cols),
                    "n_cells": int(len(y)),
                    "n_wt": int((y == 0).sum()),
                    "n_mutant": int((y == 1).sum()),
                    "cv_accuracy": acc,
                    "cv_roc_auc": auc,
                }
            )
    return {"panels": panels, "metrics": pd.DataFrame(rows), "selection": selection}


def plot_binary_panels(result: Dict, out_stem: Optional[Path] = None) -> Path:
    configure_mpl()
    out_stem = out_stem or (OUTPUT_DIR / "fig4d_binary_classification")
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    panels = result["panels"]

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.6))
    fig.suptitle(
        f"WT versus {BINARY_MUTANT_LABEL} classification "
        f"({CV_SPLITS}-fold cross-validation)",
        fontsize=14,
        y=0.98,
    )
    for i, family in enumerate(("expert", "morphagent")):
        for j, modality in enumerate(("SR", "WF")):
            ax = axes[i, j]
            p = panels[(family, modality)]
            z1, z2, y = p["z1"], p["z2"], p["y"]
            ax.scatter(z1[y == 0], z2[y == 0], s=42, c=COLOR_WT, alpha=0.88,
                       edgecolors="white", linewidths=0.3, label="WT")
            ax.scatter(z1[y == 1], z2[y == 1], s=42, c=COLOR_MUT, alpha=0.88,
                       edgecolors="white", linewidths=0.3, label="Mutant")
            ax.axvline(0.0, color="black", ls="--", lw=1.6, label="Decision line")
            ax.text(
                0.03, 0.97,
                f"Acc={p['acc']:.3f}\nAUC={p['auc']:.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          alpha=0.92, edgecolor="#666"),
            )
            if i == 0:
                ax.set_title(MODALITY_LABEL[modality], fontsize=13, pad=8)
            if j == 0:
                ax.set_ylabel(f"{FAMILY_LABEL[family]}\nComponent 2", fontsize=12)
            else:
                ax.set_ylabel("Component 2", fontsize=12)
            if i == 1:
                ax.set_xlabel("Component 1", fontsize=12)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(length=0)
            if (i, j) == (0, 0):
                ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {rel(Path(str(out_stem)))}.png/.svg/.pdf")
    return Path(f"{out_stem}.png")


# ---------------------------------------------------------------------------
# Figure 4i — six-class genotype discrimination
# ---------------------------------------------------------------------------

def sixclass_subset(df: pd.DataFrame, modality: str) -> Tuple[pd.DataFrame, np.ndarray]:
    class_to_int = {c: i for i, c in enumerate(SIX_CLASSES)}
    ids = df["sample_id"].astype(str)
    mask = ids.map(
        lambda s: modality_from_id(s) == modality and genotype_from_id(s) in class_to_int
    )
    sub = df.loc[mask].reset_index(drop=True)
    y = sub["sample_id"].astype(str).map(lambda s: class_to_int[genotype_from_id(s)]).to_numpy(int)
    return sub, y


def macro_ovr_fpr(cm: np.ndarray) -> float:
    cm = np.asarray(cm, dtype=np.int64)
    fprs = []
    for k in range(cm.shape[0]):
        fp = int(cm[:, k].sum() - cm[k, k])
        tn = int(cm.sum() - cm[k, :].sum() - cm[:, k].sum() + cm[k, k])
        if fp + tn > 0:
            fprs.append(fp / (fp + tn))
    return float(np.mean(fprs)) if fprs else float("nan")


def run_sixclass(tables: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    n_class = len(SIX_CLASSES)
    fold_rows, summary_rows = [], []
    confusion: Dict[Tuple[str, str], np.ndarray] = {}
    out_of_fold: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}

    for family in ("expert", "morphagent"):
        df = tables[family]
        cols = feature_columns(df)
        for modality in ("SR", "WF"):
            sub, y = sixclass_subset(df, modality)
            X = _design_matrix(sub, cols)
            skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            cm_total = np.zeros((n_class, n_class), dtype=np.int64)
            oof_pred = np.full(len(y), -1, dtype=int)
            for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
                clf = _logreg(multiclass=True)
                clf.fit(X[tr], y[tr])
                pred = clf.predict(X[te])
                oof_pred[te] = pred
                cm_fold = confusion_matrix(y[te], pred, labels=np.arange(n_class))
                cm_total += cm_fold
                fold_rows.append(
                    {
                        "feature_family": FAMILY_LABEL[family],
                        "modality": modality,
                        "fold": fold,
                        "accuracy": accuracy_score(y[te], pred),
                        "macro_fpr": macro_ovr_fpr(cm_fold),
                    }
                )
            confusion[(family, modality)] = cm_total
            out_of_fold[(family, modality)] = {
                "y_true": y,
                "y_pred": oof_pred,
                "sample_id": sub["sample_id"].astype(str).to_numpy(),
            }
            folds = [r for r in fold_rows
                     if r["feature_family"] == FAMILY_LABEL[family] and r["modality"] == modality]
            summary_rows.append(
                {
                    "feature_family": FAMILY_LABEL[family],
                    "family_key": family,
                    "modality": modality,
                    "n_features": len(cols),
                    "n_cells": int(len(y)),
                    "accuracy": float(np.mean([r["accuracy"] for r in folds])),
                    "macro_fpr": float(np.mean([r["macro_fpr"] for r in folds])),
                }
            )
    return {
        "summary": pd.DataFrame(summary_rows),
        "folds": pd.DataFrame(fold_rows),
        "confusion": confusion,
        "out_of_fold": out_of_fold,
    }


def _bar_order(summary: pd.DataFrame) -> pd.DataFrame:
    idx = {(r.family_key, r.modality): i for i, r in enumerate(summary.itertuples())}
    return summary.iloc[[idx[k] for k in BAR_ORDER]].reset_index(drop=True)


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


BAR_ORDER = [("expert", "SR"), ("expert", "WF"), ("morphagent", "SR"), ("morphagent", "WF")]
SIG_PAIRS = ((0, 2), (1, 2), (2, 3))


def _metric_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(SIX_CLASSES)))
    return macro_ovr_fpr(cm)


def cell_key(sample_id: str) -> str:
    """Drop the modality prefix so the SR and WF view of one cell share a key."""
    s = str(sample_id)
    return s[3:] if s[:3].lower() in {"sr_", "wf_"} else s


def sixclass_significance(
    out_of_fold: Dict[Tuple[str, str], Dict[str, np.ndarray]],
    metric: str,
    n_bootstrap: int = 2000,
    seed: int = RANDOM_STATE,
) -> Dict[Tuple[int, int], Dict[str, object]]:
    """Paired stratified bootstrap over the out-of-fold predictions.

    Every cell is predicted exactly once, in its held-out fold. Because each
    cell was imaged in both modalities, a comparison can be paired on the cell
    itself: we resample cells within genotype and evaluate both arms on the
    same resampled cells, then read the p-value off the difference
    distribution.
    """
    rng = np.random.default_rng(seed)
    lookup = {}
    for key, pack in out_of_fold.items():
        keys = np.array([cell_key(s) for s in pack["sample_id"]])
        lookup[key] = {
            "true": dict(zip(keys, pack["y_true"])),
            "pred": dict(zip(keys, pack["y_pred"])),
        }

    out: Dict[Tuple[int, int], Dict[str, object]] = {}
    for a, b in SIG_PAIRS:
        arm_a, arm_b = BAR_ORDER[a], BAR_ORDER[b]
        shared = sorted(set(lookup[arm_a]["true"]) & set(lookup[arm_b]["true"]))
        y_ref = np.array([lookup[arm_a]["true"][k] for k in shared])
        pred_a = np.array([lookup[arm_a]["pred"][k] for k in shared])
        pred_b = np.array([lookup[arm_b]["pred"][k] for k in shared])
        class_idx = [np.where(y_ref == cls)[0] for cls in np.unique(y_ref)]

        diffs = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in class_idx])
            diffs[i] = _metric_from_predictions(y_ref[idx], pred_a[idx], metric) - (
                _metric_from_predictions(y_ref[idx], pred_b[idx], metric)
            )
        p = min(1.0, 2.0 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0))))
        p = max(p, 1.0 / n_bootstrap)  # bootstrap resolution floor
        out[(a, b)] = {
            "p": p,
            "stars": _stars(p),
            "mean_diff": float(np.mean(diffs)),
            "n_paired_cells": int(len(shared)),
        }
    return out


def plot_sixclass(
    result: Dict, out_stem: Optional[Path] = None, n_bootstrap: int = 2000
) -> Tuple[Path, pd.DataFrame]:
    configure_mpl()
    out_stem = out_stem or (OUTPUT_DIR / "fig4i_sixclass_genotype")
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    summary = _bar_order(result["summary"])
    labels = [
        f"{'Expert' if r.family_key == 'expert' else 'MorphAgent'} + {r.modality}"
        for r in summary.itertuples()
    ]
    colors = [BAR_COLORS[(r.family_key, r.modality)] for r in summary.itertuples()]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    fig.suptitle(
        f"Six-class Tau genotype classification ({CV_SPLITS}-fold cross-validation)",
        fontsize=14, y=1.02,
    )
    fig.text(
        0.5, 0.955,
        "Genotypes: " + ", ".join(SIX_CLASS_LABELS),
        ha="center", fontsize=10, color="#777777",
    )

    sig_rows = []
    for ax, metric, title, arrow in (
        (axes[0], "accuracy", "Accuracy", "↑"),
        (axes[1], "macro_fpr", "macro-FPR", "↓"),
    ):
        vals = summary[metric].to_numpy(float)
        x = np.arange(len(vals))
        ax.bar(x, vals, color=colors, width=0.7, zorder=2)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=11)

        if n_bootstrap and result.get("out_of_fold"):
            sig = sixclass_significance(result["out_of_fold"], metric, n_bootstrap=n_bootstrap)
            for pair in SIG_PAIRS:
                info = sig[pair]
                sig_rows.append(
                    {
                        "metric": title,
                        "group_a": labels[pair[0]],
                        "group_b": labels[pair[1]],
                        "n_paired_cells": info["n_paired_cells"],
                        "mean_difference": info["mean_diff"],
                        "p_bootstrap": info["p"],
                        "stars": info["stars"],
                    }
                )

        ax.set_title(f"{title}  {arrow}", fontsize=13, pad=10)
        ax.set_ylabel(title, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=11)
        ax.set_ylim(0, max(0.62, float(vals.max()) + 0.08))
        ax.grid(axis="y", color="#e8e8e8", lw=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {rel(Path(str(out_stem)))}.png/.svg/.pdf")
    return Path(f"{out_stem}.png"), pd.DataFrame(sig_rows)


def confusion_table(result: Dict, family: str, modality: str) -> pd.DataFrame:
    cm = result["confusion"][(family, modality)]
    return pd.DataFrame(cm, index=SIX_CLASS_LABELS, columns=SIX_CLASS_LABELS)


# ---------------------------------------------------------------------------
# Cross-checks against the published numbers
# ---------------------------------------------------------------------------

def reference_binary_metrics() -> Optional[pd.DataFrame]:
    """Published values for the binary panel, from source/cached_results."""
    path = CACHED_RESULTS / "binary_classification_metrics.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


def reference_sixclass_metrics() -> Optional[pd.DataFrame]:
    """Published values for the six-class panel: accuracy plus fold-mean macro-FPR."""
    acc_path = CACHED_RESULTS / "sixclass_accuracy_summary.csv"
    fold_path = CACHED_RESULTS / "sixclass_fold_metrics.csv"
    if not acc_path.is_file() or not fold_path.is_file():
        return None
    acc = pd.read_csv(acc_path).rename(columns={"multiclass_cv_accuracy_mean": "accuracy"})
    fpr = (
        pd.read_csv(fold_path)
        .groupby(["feature_family", "modality"], as_index=False)["macro_fpr"]
        .mean()
    )
    ref = acc.merge(fpr, on=["feature_family", "modality"], how="left")
    return ref[["feature_family", "modality", "n_features", "accuracy", "macro_fpr"]]


def compare(reproduced: pd.DataFrame, reference: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    merged = reproduced.merge(
        reference, on=["feature_family", "modality"], suffixes=("_reproduced", "_published")
    )
    for col in cols:
        merged[f"{col}_abs_diff"] = (
            merged[f"{col}_reproduced"] - merged[f"{col}_published"]
        ).abs()
        merged[f"{col}_match_3dp"] = merged[f"{col}_abs_diff"] < 5e-4
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=TOP_K_MORPHAGENT)
    parser.add_argument("--only", choices=["binary", "sixclass"], default=None)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = load_cohort()
    print(cohort_overview(tables).to_string(index=False))

    if args.only in (None, "binary"):
        print("\n" + "=" * 72)
        print(f"Figure 4d — WT vs {BINARY_MUTANT_LABEL}")
        print("=" * 72)
        res = run_binary_panels(tables, top_k=args.top_k)
        res["metrics"].to_csv(OUTPUT_DIR / "fig4d_binary_metrics.csv", index=False)
        res["selection"].to_csv(OUTPUT_DIR / "fig4d_morphagent_top50_features.csv", index=False)
        plot_binary_panels(res)
        print(res["metrics"].to_string(index=False))
        ref = reference_binary_metrics()
        if ref is not None:
            cmp_df = compare(
                res["metrics"].rename(
                    columns={"cv_accuracy": "cv_accuracy_mean", "cv_roc_auc": "cv_roc_auc_mean"}
                ),
                ref,
                ["cv_accuracy_mean", "cv_roc_auc_mean"],
            )
            print("\nAgainst the published values:")
            print(cmp_df.filter(regex="feature_family|modality|match_3dp|_abs_diff").to_string(index=False))

    if args.only in (None, "sixclass"):
        print("\n" + "=" * 72)
        print("Figure 4i — six-class genotype discrimination")
        print("=" * 72)
        res6 = run_sixclass(tables)
        res6["summary"].to_csv(OUTPUT_DIR / "fig4i_sixclass_summary.csv", index=False)
        res6["folds"].to_csv(OUTPUT_DIR / "fig4i_sixclass_folds.csv", index=False)
        _, sig_df = plot_sixclass(res6)
        sig_df.to_csv(OUTPUT_DIR / "fig4i_sixclass_significance.csv", index=False)
        print(res6["summary"].to_string(index=False))
        print("\nBootstrap comparisons on out-of-fold predictions:")
        print(sig_df.to_string(index=False))
        ref6 = reference_sixclass_metrics()
        if ref6 is not None:
            cmp6 = compare(res6["summary"], ref6, ["accuracy", "macro_fpr"])
            print("\nAgainst the published values:")
            print(cmp6.filter(regex="feature_family|modality|match_3dp|_abs_diff").to_string(index=False))


if __name__ == "__main__":
    main()
