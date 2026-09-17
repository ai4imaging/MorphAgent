#!/usr/bin/env python3
"""Figure 5d / 5e — hierarchical morphology–gene correspondence.

Top level (5d). Each of the 11 morphology-defined biological processes has a
gene set (the union of transcripts correlated with its member features). Each
enriched GO term has its member genes. Correspondence is the shared-gene
count. Nine mitochondrial GO terms are merged into one column; two redundant
terms are dropped for display, giving an 11 × 11 matrix.

Middle level (5e). Two boxed top-level pairs are expanded into
subcellular-architecture × GRN-module maps:

  * Tau-positive object maturation × ribonucleoprotein complex biogenesis
  * Tau mislocalization × spliceosomal complex

Each architecture's gene set is the union of genes correlated with its
features; each GRN module is a TF plus its top-decile targets.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, Rectangle

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from paths import (
    CACHED_PANELD_COUNTS,
    HEATMAP_ARCHITECTURES,
    HEATMAP_FEATURE_GENES,
    HEATMAP_GO_TERMS,
    HEATMAP_GRN_MODULES,
    HEATMAP_MORPHOLOGY_CLASSES,
    OUTPUT_DIR,
    rel,
    require,
)

DROP_GO = {"endonuclease complex", "lytic vacuole membrane"}

GO_SHORT = {
    "Mitochondrion-associated (9 GO terms merged)": "Mitochondrion",
    "mRNA processing": "mRNA processing",
    "ribonucleoprotein complex biogenesis": "ribonucleoprotein complex biogenesis",
    "negative regulation of cysteine-type endopeptidase activity involved in apoptotic process":
        "Neg. reg. of caspase activity",
    "ribosome biogenesis": "ribosome biogenesis",
    "endoribonuclease complex": "endoribonuclease complex",
    "cytosolic ribosome": "cytosolic ribosome",
    "spliceosomal complex": "spliceosomal complex",
    "multi-pass translocon complex": "multi-pass translocon complex",
    "lysosomal membrane": "lysosomal membrane",
    "ribosome": "ribosome",
}

# (row index 0-based, GO_term, edge color) — boxed cells in the published panel.
HIGHLIGHTS = [
    (0, "mRNA processing", "#2E8B57"),
    (5, "ribonucleoprotein complex biogenesis", "#2F6FAE"),
    (0, "spliceosomal complex", "#C45C8A"),
    (5, "lysosomal membrane", "#C45C8A"),
]

COUNT_CMAP = LinearSegmentedColormap.from_list(
    "tau_count",
    [
        "#FCF8EC",
        "#FEE6C4",
        "#FDCF99",
        "#F9A873",
        "#F37650",
        "#DF3E2B",
        "#B71F25",
        "#7D1416",
    ],
)

TF_FACE = "#8C9DAE"
TF_EDGE = "#5E6E7D"
TGT_FACE = "#EEF1F3"
TGT_EDGE = "#9AA7B1"
TGT_TXT = "#3A434B"
EDGE_COL = "#B3BBC2"

# Display names and row order for the two middle-level panels.
MIDDLE_JOBS = [
    {
        "bp_id": "BP06",
        "bp_display": "Tau-object maturation",
        "go_term": "ribonucleoprotein complex biogenesis",
        "title": "GRNs in ribonucleoprotein complex biogenesis",
        "ylabel": "Tau-object maturation\n(Tau mislocalization)",
        "palette": ["#C58A8A", "#9AA0A6", "#C79A6B", "#8D6E4F"],
        "legend": [
            ("Small puncta, granules, foci, condensate-like objects", "#C58A8A"),
            ("Cytoplasmic, somatic, and intracellular-region", "#9AA0A6"),
            ("Boundary", "#C79A6B"),
            ("Composite objects", "#8D6E4F"),
        ],
        "sa_order": [
            "Puncta / granules / foci",
            "Cytoplasm / soma",
            "Boundary / cortex",
            "Composite objects",
        ],
        "sa_display": {
            "Puncta / granules / foci":
                "Small puncta, granules, foci, condensate-like objects",
            "Cytoplasm / soma": "Cytoplasmic, somatic, and intracellular-region",
            "Boundary / cortex": "Boundary",
            "Composite objects": "Composite objects",
        },
    },
    {
        "bp_id": "BP01",
        "bp_display": "Tau mislocalization",
        "go_term": "spliceosomal complex",
        "title": "GRNs in spliceosomal complex",
        "ylabel": "Subcellular architecture\n(Tau mislocalization)",
        "palette": ["#73AEA8", "#7E9BB3", "#9B8FBE", "#C79A6B"],
        "legend": [
            ("Nuclear and perinuclear Tau architecture", "#73AEA8"),
            ("Nuclear-cytoplasmic compartments", "#7E9BB3"),
            ("Boundary", "#9B8FBE"),
            ("Whole-cell geometry", "#C79A6B"),
        ],
        "sa_order": [
            "Perinuclear ring",
            "Nuclear–cytoplasmic compartments",
            "Boundary / cortex",
            "Whole-cell geometry",
        ],
        "sa_display": {
            "Perinuclear ring": "Nuclear and perinuclear Tau architecture",
            "Nuclear–cytoplasmic compartments": "Nuclear-cytoplasmic compartments",
            "Boundary / cortex": "Boundary",
            "Whole-cell geometry": "Whole-cell geometry",
        },
    },
]


def configure_mpl() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["axes.linewidth"] = 0.6


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


def _display_png(path: Path) -> None:
    """Embed the saved PNG in the calling notebook cell.

    Cursor / VS Code notebooks often ignore Image(filename=...); passing the
    bytes makes the figure show up as cell output.
    """
    if not _in_notebook():
        return
    try:
        from IPython.display import Image, display
        display(Image(data=Path(path).read_bytes(), format="png"))
    except Exception:
        return


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def parse_genes(value: object) -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    text = str(value).replace("/", ";").replace(",", ";")
    return {g.strip().upper() for g in text.split(";") if g.strip()}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_morphology_classes() -> pd.DataFrame:
    df = pd.read_csv(require(HEATMAP_MORPHOLOGY_CLASSES))
    df["gene_set"] = df["genes"].map(parse_genes)
    return df


def load_go_terms(drop_redundant: bool = True) -> pd.DataFrame:
    df = pd.read_csv(require(HEATMAP_GO_TERMS))
    if drop_redundant:
        df = df[~df["go_display_label"].isin(DROP_GO)].copy()
        df = df.sort_values("go_display_order").reset_index(drop=True)
        df["go_display_order"] = np.arange(1, len(df) + 1)
    df["gene_set"] = df["genes"].map(parse_genes)
    df["go_short"] = df["go_display_label"].map(GO_SHORT).fillna(df["go_display_label"])
    return df


def load_feature_genes() -> Dict[str, set[str]]:
    df = pd.read_csv(require(HEATMAP_FEATURE_GENES))
    return {str(r.feature): parse_genes(r.genes) for r in df.itertuples(index=False)}


def load_architectures(bp_id: str) -> pd.DataFrame:
    df = pd.read_csv(require(HEATMAP_ARCHITECTURES))
    sub = df[df["bp_id"] == bp_id].copy()
    return sub.reset_index(drop=True)


def load_grns(go_term: str) -> pd.DataFrame:
    df = pd.read_csv(require(HEATMAP_GRN_MODULES))
    sub = df[df["GO_term"] == go_term].copy()
    sub["grn_order"] = sub["GRN_id"].str.replace("GRN", "", regex=False).astype(int)
    return sub.sort_values("grn_order").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Top-level overlap
# ---------------------------------------------------------------------------

def top_level_overlap(
    morph: Optional[pd.DataFrame] = None,
    go: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    morph = load_morphology_classes() if morph is None else morph
    go = load_go_terms() if go is None else go
    rows = []
    for _, m in morph.iterrows():
        for _, g in go.iterrows():
            shared = sorted(m["gene_set"] & g["gene_set"])
            rows.append({
                "bp_id": m["bp_id"],
                "morphology_class_order": int(m["morphology_class_order"]),
                "morphology_class": m["morphology_class"],
                "bp_display": m["bp_display"],
                "GO_display_order": int(g["go_display_order"]),
                "GO_term": g["go_display_label"],
                "GO_short": g["go_short"],
                "n_morph_genes": len(m["gene_set"]),
                "n_GO_genes": len(g["gene_set"]),
                "shared_gene_count": len(shared),
                "shared_genes": ";".join(shared) if shared else "",
            })
    return pd.DataFrame(rows)


def top_level_matrix(long_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = (
        long_df[["morphology_class_order", "bp_display"]]
        .drop_duplicates()
        .sort_values("morphology_class_order")
    )
    cols = (
        long_df[["GO_display_order", "GO_term", "GO_short"]]
        .drop_duplicates()
        .sort_values("GO_display_order")
    )
    pivot = long_df.pivot(
        index="morphology_class_order",
        columns="GO_display_order",
        values="shared_gene_count",
    ).loc[rows["morphology_class_order"], cols["GO_display_order"]]
    return pivot, rows, cols


def plot_top_level_heatmap(
    long_df: pd.DataFrame,
    out_path: Optional[Path] = None,
) -> Path:
    configure_mpl()
    out_path = Path(out_path) if out_path else OUTPUT_DIR / "fig5d_bp_go_count_heatmap.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pivot, rows, cols = top_level_matrix(long_df)
    data = pivot.values.astype(float)
    n_rows, n_cols = data.shape
    vmax = int(np.nanmax(data))

    cell_w, cell_h = 0.58, 0.36
    left_lab, top_lab = 2.55, 2.85
    right_pad, bottom_pad, title_pad = 1.20, 0.35, 0.50
    grid_w, grid_h = n_cols * cell_w, n_rows * cell_h
    fig_w = left_lab + grid_w + right_pad
    fig_h = bottom_pad + grid_h + top_lab + title_pad

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([
        left_lab / fig_w, bottom_pad / fig_h, grid_w / fig_w, grid_h / fig_h,
    ])
    norm = Normalize(0, vmax)
    im = ax.imshow(
        data, cmap="OrRd", norm=norm, aspect="auto",
        origin="upper", extent=[0, n_cols, n_rows, 0],
    )
    for x in range(n_cols + 1):
        ax.axvline(x, color="white", lw=0.8)
    for y in range(n_rows + 1):
        ax.axhline(y, color="white", lw=0.8)
    for sp in ax.spines.values():
        sp.set_color("#7a7a7a")
        sp.set_linewidth(0.8)

    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            txt = "white" if norm(v) >= 0.60 else "#222222"
            ax.text(
                j + 0.5, i + 0.5, f"{int(v)}",
                ha="center", va="center", fontsize=8.0, color=txt,
            )

    go_to_j = {g: j for j, g in enumerate(cols["GO_term"].tolist())}
    for i, go, color in HIGHLIGHTS:
        j = go_to_j[go]
        ax.add_patch(Rectangle(
            (j, i), 1, 1, fill=False, edgecolor=color, linewidth=2.2, zorder=5,
        ))

    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_xticklabels(
        [wrap(c, 18) for c in cols["GO_short"]],
        rotation=90, ha="center", va="bottom", fontsize=7.0, linespacing=0.95,
    )
    ax.xaxis.set_ticks_position("top")
    ax.set_yticklabels(
        [wrap(r, 28) for r in rows["bp_display"]],
        fontsize=7.5, va="center", linespacing=0.95,
    )
    ax.tick_params(length=0)
    ax.set_ylabel(
        "Morphology-defined biological processes",
        fontsize=9, fontweight="bold", labelpad=10,
    )
    ax.set_xlabel("GO-enriched biological processes", fontsize=9, fontweight="bold", labelpad=8)
    ax.xaxis.set_label_position("top")

    fig.text(
        0.02, 0.97, "d", ha="left", va="top", fontsize=16, fontweight="bold",
        transform=fig.transFigure,
    )
    fig.text(
        (left_lab + grid_w / 2) / fig_w, 0.998,
        "BP-level morphology–GO correspondence",
        ha="center", va="top", fontsize=11, fontweight="bold",
    )

    cax = fig.add_axes([
        (left_lab + grid_w + 0.22) / fig_w, bottom_pad / fig_h,
        0.16 / fig_w, grid_h / fig_h,
    ])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#7a7a7a")
    cb.set_label("Count of shared genes", fontsize=8, fontweight="bold")
    cb.set_ticks(list(range(0, vmax + 1, max(1, vmax // 4))))
    cb.ax.tick_params(labelsize=7, length=2)

    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    fig.savefig(out_path.with_suffix(".svg"), format="svg", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    _display_png(out_path)
    print("wrote", rel(out_path))
    return out_path


def compare_top_level(long_df: pd.DataFrame) -> pd.DataFrame:
    published = pd.read_csv(require(CACHED_PANELD_COUNTS))
    merged = long_df.merge(
        published[
            ["morphology_class_order", "GO_term", "shared_gene_count"]
        ].rename(columns={"shared_gene_count": "published_count"}),
        on=["morphology_class_order", "GO_term"],
        how="left",
    )
    merged["match"] = merged["shared_gene_count"] == merged["published_count"]
    return merged[[
        "bp_display", "GO_short", "shared_gene_count", "published_count", "match", "shared_genes",
    ]]


# ---------------------------------------------------------------------------
# Middle-level overlap
# ---------------------------------------------------------------------------

def _match_sa_name(name: str, candidates: Iterable[str]) -> str:
    compact = name.replace("–", "-").replace("—", "-")
    for c in candidates:
        if c.replace("–", "-").replace("—", "-") == compact:
            return c
    raise KeyError(f"Architecture {name!r} not in {list(candidates)}")


def architecture_gene_sets(
    bp_id: str,
    feat2genes: Optional[Dict[str, set[str]]] = None,
) -> List[dict]:
    feat2genes = load_feature_genes() if feat2genes is None else feat2genes
    arch = load_architectures(bp_id)
    rows = []
    for _, r in arch.iterrows():
        feats = [f for f in str(r["features"]).split(";") if f]
        genes: set[str] = set()
        for f in feats:
            genes |= feat2genes.get(f, set())
        rows.append({
            "bp_id": bp_id,
            "sa_id": r["sa_id"],
            "sa_local_id": r["sa_local_id"],
            "architecture": r["sa_en"],
            "n_features": len(feats),
            "features": ";".join(feats),
            "n_genes": len(genes),
            "genes": ";".join(sorted(genes)),
            "gene_set": genes,
        })
    return rows


def middle_level_overlap(
    job: dict,
    feat2genes: Optional[Dict[str, set[str]]] = None,
) -> pd.DataFrame:
    sas = architecture_gene_sets(job["bp_id"], feat2genes)
    by_name = {s["architecture"]: s for s in sas}
    grns = load_grns(job["go_term"])
    rows = []
    for order, sa_name in enumerate(job["sa_order"], start=1):
        key = _match_sa_name(sa_name, by_name)
        sa = by_name[key]
        for _, grn in grns.iterrows():
            gene_set = parse_genes(grn["genes"])
            shared = sorted(sa["gene_set"] & gene_set)
            rows.append({
                "bp_id": job["bp_id"],
                "bp_display": job["bp_display"],
                "GO_term": job["go_term"],
                "architecture_order": order,
                "architecture": sa["architecture"],
                "architecture_display": job["sa_display"][sa_name],
                "n_architecture_genes": sa["n_genes"],
                "GRN_id": grn["GRN_id"],
                "GRN_TF": grn["TF"],
                "GRN_targets": grn["targets"],
                "n_GRN_genes": int(grn["n_genes"]),
                "shared_gene_count": len(shared),
                "shared_genes": ";".join(shared) if shared else "",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# GRN star diagrams + middle-level heatmaps
# ---------------------------------------------------------------------------

def _module_radius(tf: str, targets: Sequence[str]) -> float:
    n = len(targets)
    max_len = max([len(tf)] + [len(t) for t in targets] or [1])
    if n == 1:
        return 0.85
    return 1.15 + 0.08 * max(0, n - 2) + 0.03 * max(0, max_len - 5)


def _draw_grn_module(ax, cx: float, cy: float, tf: str, targets: Sequence[str], scale: float = 1.0) -> None:
    n = len(targets)
    radius = _module_radius(tf, targets) * scale
    if n == 1:
        tf_pos = (cx - 0.70 * scale, cy)
        pos = [(cx + 0.70 * scale, cy)]
    else:
        tf_pos = (cx, cy)
        angles = [np.pi / 2 - 2 * np.pi * i / n for i in range(n)]
        pos = [(cx + radius * np.cos(a), cy + radius * np.sin(a)) for a in angles]

    for (x, y) in pos:
        ax.annotate(
            "", xy=(x, y), xytext=tf_pos,
            arrowprops=dict(
                arrowstyle="-|>", color=EDGE_COL, lw=1.2 * scale,
                shrinkA=16 * scale, shrinkB=13 * scale, mutation_scale=11,
            ),
            zorder=1,
        )

    fs_tgt = 7.0 if n >= 5 else 7.6
    for (x, y), name in zip(pos, targets):
        ax.text(
            x, y, name, ha="center", va="center",
            fontsize=fs_tgt, color=TGT_TXT, zorder=3,
            bbox=dict(boxstyle="round,pad=0.20", fc=TGT_FACE, ec=TGT_EDGE, lw=0.8),
            clip_on=False,
        )
    ax.text(
        tf_pos[0], tf_pos[1], tf, ha="center", va="center",
        fontsize=8.2 if len(tf) <= 6 else 7.2,
        fontweight="bold", color="white", zorder=4,
        bbox=dict(boxstyle="circle,pad=0.32", fc=TF_FACE, ec=TF_EDGE, lw=1.1),
        clip_on=False,
    )


def _draw_grn_legend(ax, x: float, y: float) -> None:
    ax.add_patch(FancyBboxPatch(
        (x, y - 0.16), 0.52, 0.32,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=TGT_FACE, edgecolor=TGT_EDGE, lw=0.8, zorder=5,
    ))
    ax.text(x + 0.62, y, "Target", ha="left", va="center", fontsize=7.2, color="#444", zorder=6)
    ax.scatter([x + 1.55], [y], s=150, facecolor=TF_FACE, edgecolor=TF_EDGE, lw=1.0, zorder=5)
    ax.text(x + 1.55, y, "TF", ha="center", va="center", fontsize=6.2, fontweight="bold", color="white", zorder=6)
    ax.text(x + 1.78, y, "TF", ha="left", va="center", fontsize=7.2, color="#444")


def _draw_grn_strip(ax, grns: pd.DataFrame) -> None:
    ax.set_aspect("equal")
    ax.axis("off")
    modules = []
    for _, r in grns.iterrows():
        targets = [t for t in str(r["targets"]).split(";") if t]
        modules.append((r["TF"], targets))
    n = len(modules)
    ncols = 4 if n > 4 else max(n, 1)
    nrows = int(np.ceil(n / ncols))
    extents = [_module_radius(tf, tgts) + 1.15 for tf, tgts in modules]
    cell = max(extents) * 2.25 if extents else 3.4
    for idx, (tf, targets) in enumerate(modules):
        r = idx // ncols
        c = idx % ncols
        cx = (c + 0.5) * cell
        cy = (nrows - 1 - r + 0.5) * cell
        _draw_grn_module(ax, cx, cy, tf, targets, scale=0.92)
    ax.set_xlim(-0.10, ncols * cell + 0.10)
    ax.set_ylim(-0.15, nrows * cell + 0.55)
    _draw_grn_legend(ax, ncols * cell - 2.45, nrows * cell + 0.28)


def _draw_count_heatmap(fig, box, long_df: pd.DataFrame, job: dict, draw_cbar: bool) -> None:
    row_order = (
        long_df[["architecture_order", "architecture_display"]]
        .drop_duplicates()
        .sort_values("architecture_order")
    )
    col_order = sorted(long_df["GRN_id"].unique(), key=lambda x: int(x.replace("GRN", "")))
    pivot = long_df.pivot(
        index="architecture_order", columns="GRN_id", values="shared_gene_count",
    ).loc[row_order["architecture_order"], col_order]
    data = np.clip(pivot.values.astype(float), 0, 2)
    n_rows, n_cols = data.shape
    palette = job["palette"][:n_rows]

    left, bottom, width, height = box
    ylabel_frac = 0.045
    label_frac = 0.38
    swatch_frac = 0.048
    gap_frac = 0.016
    cbar_frac = 0.10 if draw_cbar else 0.0
    heat_frac = 1 - ylabel_frac - label_frac - swatch_frac - gap_frac - cbar_frac
    x_lab = left + width * ylabel_frac
    x_swatch = x_lab + width * label_frac
    x_heat = x_swatch + width * (swatch_frac + gap_frac)
    heat_w = width * heat_frac

    fig.text(
        left + width * 0.012, bottom + height * 0.50, job["ylabel"],
        rotation=90, ha="center", va="center", fontsize=8.2, fontweight="bold",
    )

    ax_lab = fig.add_axes([x_lab, bottom, width * label_frac, height])
    ax_lab.set_xlim(0, 1)
    ax_lab.set_ylim(n_rows, 0)
    ax_lab.axis("off")
    for i, lab in enumerate(row_order["architecture_display"]):
        ax_lab.text(0.98, i + 0.5, wrap(lab, 26), ha="right", va="center", fontsize=7.2, linespacing=0.95)

    ax_s = fig.add_axes([x_swatch, bottom, width * swatch_frac, height])
    ax_s.imshow(
        np.arange(n_rows).reshape(-1, 1),
        cmap=ListedColormap(palette), aspect="auto", extent=[0, 1, n_rows, 0],
    )
    for y in range(n_rows + 1):
        ax_s.axhline(y, color="white", lw=1.0)
    ax_s.set_xticks([])
    ax_s.set_yticks([])
    for sp in ax_s.spines.values():
        sp.set_color("#7a7a7a")
        sp.set_linewidth(0.8)

    ax = fig.add_axes([x_heat, bottom, heat_w, height])
    im = ax.imshow(
        data, cmap=COUNT_CMAP, norm=Normalize(0, 2), aspect="auto",
        origin="upper", extent=[0, n_cols, n_rows, 0],
    )
    for x in range(n_cols + 1):
        ax.axvline(x, color="white", lw=0.8)
    for y in range(n_rows + 1):
        ax.axhline(y, color="white", lw=0.8)
    for sp in ax.spines.values():
        sp.set_color("#7a7a7a")
        sp.set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])

    if draw_cbar:
        cax = fig.add_axes([x_heat + heat_w + width * 0.02, bottom, width * 0.028, height])
        cb = fig.colorbar(im, cax=cax)
        cb.outline.set_linewidth(0.6)
        cb.outline.set_edgecolor("#7a7a7a")
        cb.set_label("Count of shared genes", fontsize=7.5, fontweight="bold")
        cb.set_ticks([0, 1, 2])
        cb.ax.tick_params(labelsize=7, length=2)
    return im


def _draw_legend(ax, items: Sequence[Tuple[str, str]]) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    n = len(items)
    for i, (label, color) in enumerate(items):
        x = 0.02 + (i % 2) * 0.50
        y = 0.72 if i < 2 else 0.28
        ax.add_patch(Rectangle((x, y - 0.12), 0.045, 0.24, facecolor=color, edgecolor="#7a7a7a", lw=0.4))
        ax.text(x + 0.06, y, label, ha="left", va="center", fontsize=7.0)


def plot_middle_level_panel(
    tables: Dict[str, pd.DataFrame],
    out_path: Optional[Path] = None,
) -> Path:
    configure_mpl()
    out_path = Path(out_path) if out_path else OUTPUT_DIR / "fig5e_architecture_grn_heatmap.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(17.4, 8.4), facecolor="white")
    fig.text(0.012, 0.975, "e", ha="left", va="top", fontsize=16, fontweight="bold")

    col_left = [0.03, 0.535]
    col_width = [0.49, 0.45]
    for idx, job in enumerate(MIDDLE_JOBS):
        long_df = tables[job["bp_id"]]
        grns = load_grns(job["go_term"])
        x0, w = col_left[idx], col_width[idx]
        fig.text(
            x0 + w * 0.52, 0.965, job["title"],
            ha="center", va="top", fontsize=11.5, fontweight="bold",
        )
        ax_grn = fig.add_axes([x0, 0.44, w * 0.98, 0.49])
        _draw_grn_strip(ax_grn, grns)
        _draw_count_heatmap(
            fig, [x0, 0.155, w * 0.98, 0.255], long_df, job, draw_cbar=True,
        )
        ax_leg = fig.add_axes([x0, 0.012, w * 0.98, 0.12])
        _draw_legend(ax_leg, job["legend"])

    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    fig.savefig(out_path.with_suffix(".svg"), format="svg", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    _display_png(out_path)
    print("wrote", rel(out_path))
    return out_path


def highlight_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, go, color in HIGHLIGHTS:
        hit = long_df[
            (long_df["morphology_class_order"] == i + 1) & (long_df["GO_term"] == go)
        ].iloc[0]
        rows.append({
            "bp_display": hit["bp_display"],
            "GO_short": hit["GO_short"],
            "shared_gene_count": int(hit["shared_gene_count"]),
            "shared_genes": hit["shared_genes"],
            "box": {"#2E8B57": "green", "#2F6FAE": "blue", "#C45C8A": "pink"}[color],
        })
    return pd.DataFrame(rows)


def run_all(output_dir: Optional[Path] = None) -> dict:
    output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    morph = load_morphology_classes()
    go = load_go_terms()
    top = top_level_overlap(morph, go)
    top.to_csv(output_dir / "fig5d_bp_go_overlap_long.csv", index=False)
    top_png = plot_top_level_heatmap(top, output_dir / "fig5d_bp_go_count_heatmap.png")

    feat2genes = load_feature_genes()
    middle = {}
    for job in MIDDLE_JOBS:
        df = middle_level_overlap(job, feat2genes)
        slug = job["bp_id"].lower()
        df.to_csv(output_dir / f"fig5e_{slug}_overlap_long.csv", index=False)
        middle[job["bp_id"]] = df
    mid_png = plot_middle_level_panel(middle, output_dir / "fig5e_architecture_grn_heatmap.png")

    return {
        "top": top,
        "middle": middle,
        "top_png": top_png,
        "middle_png": mid_png,
        "check": compare_top_level(top),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("top", "middle", "all"), default="all")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.only in ("top", "all"):
        top = top_level_overlap()
        top.to_csv(OUTPUT_DIR / "fig5d_bp_go_overlap_long.csv", index=False)
        path = plot_top_level_heatmap(top)
        check = compare_top_level(top)
        print(rel(path))
        print(highlight_summary(top).to_string(index=False))
        print(f"top-level cells matching published counts: {int(check['match'].sum())}/{len(check)}")

    if args.only in ("middle", "all"):
        feat2genes = load_feature_genes()
        tables = {}
        for job in MIDDLE_JOBS:
            df = middle_level_overlap(job, feat2genes)
            df.to_csv(OUTPUT_DIR / f"fig5e_{job['bp_id'].lower()}_overlap_long.csv", index=False)
            tables[job["bp_id"]] = df
            nz = df[df["shared_gene_count"] > 0]
            print(f"\n{job['bp_display']} × {job['go_term']}")
            if len(nz):
                print(nz[[
                    "architecture_display", "GRN_id", "GRN_TF",
                    "shared_gene_count", "shared_genes",
                ]].to_string(index=False))
            else:
                print("  (no overlapping genes)")
        path = plot_middle_level_panel(tables)
        print("\n" + rel(path))


if __name__ == "__main__":
    main()
