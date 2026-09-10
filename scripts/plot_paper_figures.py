#!/usr/bin/env python3
"""Generate publication figures from the final ASB paper tables.

The script uses only final paper-facing CSV files in paper_tables/ and writes
editable vector files plus high-resolution raster exports to latex/figures/.
"""

from __future__ import annotations

import os
from pathlib import Path
from textwrap import fill

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = ROOT / "latex" / "figures"
SOURCE_DIR = OUT_DIR / "source_data"
TABLE_DIR = ROOT / "paper_tables"

DATASET_ORDER = ["SMS Spam", "SST-2", "ADE", "Disaster Tweets"]
SHORT_DATASET = {
    "SMS Spam": "SMS",
    "SST-2": "SST-2",
    "ADE": "ADE",
    "Disaster Tweets": "Disaster",
}

PALETTE = {
    "ink": "#202124",
    "muted": "#6B7280",
    "grid": "#E5E7EB",
    "paper": "#FFFFFF",
    "blue": "#5B8DB8",
    "teal": "#67A9A1",
    "green": "#7BAF7A",
    "gold": "#D8A24A",
    "orange": "#C9895B",
    "red": "#C86F6F",
    "purple": "#8E7DBE",
    "slate": "#7A869A",
    "pale_blue": "#EAF2F8",
    "pale_green": "#ECF6F1",
    "pale_gold": "#FBF2DC",
}

METHOD_COLORS = {
    "TF-IDF + LR": PALETTE["slate"],
    "RoBERTa embedding + LR": PALETTE["purple"],
    "DistilBERT fine-tuning": PALETTE["blue"],
    "ASB-LR (ours)": PALETTE["teal"],
}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.3,
            "axes.linewidth": 0.55,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def clean_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=PALETTE["grid"], linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.07) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color=PALETTE["ink"],
    )


def save_figure(fig: plt.Figure, stem: str, *, dpi: int = 600) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.pdf")
    fig.savefig(OUT_DIR / f"{stem}.svg")
    fig.savefig(OUT_DIR / f"{stem}.tiff", dpi=dpi)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


def metric_value(value: object) -> float:
    text = str(value)
    if "±" in text:
        text = text.split("±", 1)[0]
    return float(text.strip())


def write_source_data(name: str, df: pd.DataFrame) -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SOURCE_DIR / f"{name}.csv", index=False)


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    subtitle: str,
    facecolor: str,
    edgecolor: str,
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=0.75,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    x, y = xy
    ax.text(
        x + width / 2,
        y + height * 0.62,
        title,
        ha="center",
        va="center",
        fontsize=7.1,
        fontweight="bold",
        color=PALETTE["ink"],
    )
    ax.text(
        x + width / 2,
        y + height * 0.31,
        subtitle,
        ha="center",
        va="center",
        fontsize=5.9,
        color="#4B5563",
        linespacing=1.08,
    )


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.75,
            color="#4B5563",
            shrinkA=3,
            shrinkB=3,
        )
    )


def plot_workflow() -> None:
    """Figure 1: method and diagnostic workflow."""
    fig, ax = plt.subplots(figsize=(7.1, 3.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.02,
        0.95,
        "ASB semantic feature workflow",
        fontsize=9.2,
        fontweight="bold",
        ha="left",
        va="top",
        color=PALETTE["ink"],
    )

    y = 0.58
    w = 0.145
    h = 0.21
    xs = [0.03, 0.215, 0.400, 0.585, 0.770]
    boxes = [
        ("Training texts", "500 stratified\nexamples", "#F3F4F6", PALETTE["slate"]),
        ("Question schema", "20 binary questions\n5 groups x 4", PALETTE["pale_blue"], PALETTE["blue"]),
        ("LLM encoder", "one JSON call\nper text", "#EEF7F6", PALETTE["teal"]),
        ("Feature matrix", "5000 x 20\nbinary variables", "#F0F8EC", PALETTE["green"]),
        ("Classifier", "LR / XGBoost\ntrain split only", PALETTE["pale_gold"], PALETTE["gold"]),
    ]
    for x, (title, subtitle, face, edge) in zip(xs, boxes):
        draw_box(ax, (x, y), w, h, title, subtitle, face, edge)
    for idx in range(len(xs) - 1):
        draw_arrow(ax, (xs[idx] + w, y + h / 2), (xs[idx + 1], y + h / 2))

    draw_box(
        ax,
        (0.035, 0.18),
        0.255,
        0.22,
        "Controlled protocol",
        "clean / deduplicate\nstratified 4000 / 1000\nfixed random seeds",
        "#FFFFFF",
        PALETTE["slate"],
    )
    draw_box(
        ax,
        (0.385, 0.18),
        0.520,
        0.22,
        "Reliability checks",
        "prompt stability | MI selection | group utility\nsemantic edits | extractor sensitivity | cost audit",
        "#FFFFFF",
        PALETTE["red"],
    )
    draw_arrow(ax, (0.162, 0.40), (0.162, y))
    draw_arrow(ax, (0.645, y), (0.645, 0.40))

    ax.text(
        0.02,
        0.055,
        "The LLM output is treated as a reusable data artifact: predictive value is evaluated together with stability, diagnostic transparency, and operating cost.",
        fontsize=6.6,
        color="#4B5563",
        ha="left",
        va="bottom",
    )
    save_figure(fig, "workflow_overview")


def performance_panel_data() -> pd.DataFrame:
    df = pd.read_csv(TABLE_DIR / "table_unified_method_performance.csv")
    keep = list(METHOD_COLORS)
    df = df[df["Method"].isin(keep)].copy()
    df["Macro-F1 numeric"] = df["Macro-F1"].map(metric_value)
    df["Dataset"] = pd.Categorical(df["Dataset"], DATASET_ORDER, ordered=True)
    df["Method"] = pd.Categorical(df["Method"], keep, ordered=True)
    df = df.sort_values(["Dataset", "Method"])
    write_source_data("figure2a_performance_macro_f1", df)
    return df


def prompt_panel_data() -> pd.DataFrame:
    df = pd.read_csv(TABLE_DIR / "table_prompt_stability_summary.csv")
    summary = (
        df.groupby("Dataset", as_index=False)
        .agg(
            min_matrix_agreement=("Matrix agreement", "min"),
            min_prediction_agreement=("Prediction agreement", "min"),
        )
    )
    summary["Dataset"] = pd.Categorical(summary["Dataset"], DATASET_ORDER, ordered=True)
    summary = summary.sort_values("Dataset")
    write_source_data("figure2b_prompt_stability", summary)
    return summary


def audit_panel_data() -> pd.DataFrame:
    concept = pd.read_csv(TABLE_DIR / "table_concept_audit_summary.csv")
    concept = concept[concept["Feature"] == "ALL"][
        ["Dataset", "Agreement", "Cohen kappa", "Positive agreement"]
    ].copy()
    concept = concept.rename(
        columns={
            "Agreement": "Primary human vs ASB agreement",
            "Cohen kappa": "Primary human vs ASB kappa",
            "Positive agreement": "Primary human vs ASB positive agreement",
        }
    )
    second = pd.read_csv(TABLE_DIR / "table_second_annotator_agreement.csv")
    second = second.rename(
        columns={
            "Agreement": "Human-human agreement",
            "Cohen's kappa": "Human-human kappa",
            "Positive agreement": "Human-human positive agreement",
        }
    )
    out = concept.merge(
        second[
            [
                "Dataset",
                "Human-human agreement",
                "Human-human kappa",
                "Human-human positive agreement",
            ]
        ],
        on="Dataset",
        how="left",
    )
    out["Dataset"] = pd.Categorical(out["Dataset"], DATASET_ORDER, ordered=True)
    out = out.sort_values("Dataset")
    write_source_data("main_claims_audit_reliability", out)
    return out


def mi_panel_data() -> pd.DataFrame:
    df = pd.read_csv(TABLE_DIR / "table_mi_feature_selection_sensitivity.csv")
    rows = []
    for dataset, part in df.groupby("Dataset", sort=False):
        baseline = float(part.loc[part["Feature set"] == "All ASB features", "Macro-F1"].iloc[0])
        for _, row in part.iterrows():
            feature_set = str(row["Feature set"])
            if feature_set == "All ASB features":
                k = 20
                label = "All"
            else:
                k = int(feature_set.replace("MI top-", ""))
                label = f"top-{k}"
            macro = float(row["Macro-F1"])
            rows.append(
                {
                    "Dataset": dataset,
                    "Feature set": label,
                    "#Features": k,
                    "Macro-F1": macro,
                    "Delta Macro-F1 vs all": macro - baseline,
                }
            )
    out = pd.DataFrame(rows)
    out["Dataset"] = pd.Categorical(out["Dataset"], DATASET_ORDER, ordered=True)
    out = out.sort_values(["Dataset", "#Features"])
    write_source_data("figure2c_mi_selection", out)
    return out


def counterfactual_panel_data() -> pd.DataFrame:
    df = pd.read_csv(TABLE_DIR / "table_counterfactual_feature_edit_coverage.csv")
    rows = []
    for _, row in df.iterrows():
        for col, budget in [("<=1 edit", 1), ("<=2 edits", 2), ("<=3 edits", 3)]:
            rows.append({"Dataset": row["Dataset"], "Edit budget": budget, "Coverage": float(row[col])})
    out = pd.DataFrame(rows)
    out["Dataset"] = pd.Categorical(out["Dataset"], DATASET_ORDER, ordered=True)
    out = out.sort_values(["Dataset", "Edit budget"])
    write_source_data("figure2d_counterfactual_coverage", out)
    return out


def provider_panel_data() -> pd.DataFrame:
    df = pd.read_csv(TABLE_DIR / "table_model_comparison_asb_lr.csv")
    df["Macro-F1"] = df["Macro-F1"].astype(float)
    write_source_data("figure2e_provider_comparison", df)
    return df


def group_panel_data() -> pd.DataFrame:
    df = pd.read_csv(TABLE_DIR / "table_feature_group_ablation_condensed.csv")
    df["Macro-F1"] = df["Macro-F1"].astype(float)
    rows = []
    for dataset in DATASET_ORDER:
        part = df[df["Dataset"] == dataset].copy()
        all_row = part[part["Feature group used"] == "All features"].iloc[0]
        single = part[part["Feature group used"] != "All features"].sort_values("Macro-F1", ascending=False).iloc[0]
        rows.append(
            {
                "Dataset": dataset,
                "Feature set": "All features",
                "Feature group": "All features",
                "Macro-F1": float(all_row["Macro-F1"]),
            }
        )
        rows.append(
            {
                "Dataset": dataset,
                "Feature set": "Best single group",
                "Feature group": str(single["Feature group used"]).replace("Only ", ""),
                "Macro-F1": float(single["Macro-F1"]),
            }
        )
    out = pd.DataFrame(rows)
    out["Dataset"] = pd.Categorical(out["Dataset"], DATASET_ORDER, ordered=True)
    out = out.sort_values(["Dataset", "Feature set"])
    write_source_data("figure2f_group_utility", out)
    return out


def cost_panel_data() -> pd.DataFrame:
    df = pd.read_csv(TABLE_DIR / "table_api_cost_summary.csv")
    df["Estimated cost"] = df["Estimated cost"].astype(float)
    df["Cost per 1k rows"] = df["Estimated cost"] / df["Feature rows"].astype(float) * 1000
    write_source_data("figure2f_api_cost", df)
    return df


def plot_performance(ax: plt.Axes) -> None:
    df = performance_panel_data()
    x = np.arange(len(DATASET_ORDER))
    methods = list(METHOD_COLORS)
    offsets = np.linspace(-0.27, 0.27, len(methods))
    bar_width = 0.17
    short_names = {
        "TF-IDF + LR": "TF-IDF + LR",
        "RoBERTa embedding + LR": "RoBERTa + LR",
        "DistilBERT fine-tuning": "DistilBERT-FT",
        "ASB-LR (ours)": "ASB-LR",
    }
    for idx, method in enumerate(methods):
        values = []
        for dataset in DATASET_ORDER:
            match = df[(df["Dataset"] == dataset) & (df["Method"] == method)]
            values.append(float(match["Macro-F1 numeric"].iloc[0]))
        is_asb = method == "ASB-LR (ours)"
        ax.bar(
            x + offsets[idx],
            values,
            width=bar_width,
            color=METHOD_COLORS[method],
            alpha=1.0 if is_asb else 0.74,
            edgecolor=PALETTE["ink"] if is_asb else "white",
            linewidth=0.55 if is_asb else 0.35,
            label=short_names[method],
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_DATASET[d] for d in DATASET_ORDER])
    ax.set_ylim(0.70, 1.00)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Predictive performance")
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.995, 0.985),
        frameon=False,
        ncol=2,
        borderaxespad=0.15,
        handletextpad=0.35,
        columnspacing=0.75,
    )
    clean_axis(ax)
    panel_label(ax, "a")


def plot_prompt_stability(ax: plt.Axes, label: str = "b") -> None:
    df = prompt_panel_data()
    y_pos = {dataset: len(DATASET_ORDER) - 1 - idx for idx, dataset in enumerate(DATASET_ORDER)}
    y = np.array([y_pos[str(dataset)] for dataset in df["Dataset"]])
    ax.scatter(
        df["min_matrix_agreement"],
        y + 0.10,
        s=32,
        color=PALETTE["blue"],
        edgecolor="white",
        linewidth=0.4,
        label="feature matrix",
        zorder=3,
    )
    ax.scatter(
        df["min_prediction_agreement"],
        y - 0.10,
        s=32,
        color=PALETTE["gold"],
        edgecolor="white",
        linewidth=0.4,
        label="prediction",
        zorder=3,
    )
    for _, row in df.iterrows():
        yi = y_pos[str(row["Dataset"])]
        ax.plot(
            [row["min_matrix_agreement"], row["min_prediction_agreement"]],
            [yi + 0.10, yi - 0.10],
            color=PALETTE["grid"],
            linewidth=0.8,
            zorder=1,
        )
    ax.set_yticks([y_pos[d] for d in DATASET_ORDER])
    ax.set_yticklabels([SHORT_DATASET[d] for d in DATASET_ORDER])
    ax.set_ylim(-0.45, len(DATASET_ORDER) - 0.55)
    ax.set_xlim(0.94, 1.002)
    ax.set_xlabel("Minimum pairwise agreement")
    ax.set_title("Prompt stability")
    ax.legend(loc="upper left", frameon=False, borderaxespad=0.15, handletextpad=0.35)
    clean_axis(ax, grid_axis="x")
    panel_label(ax, label)


def plot_prompt_stability_main(ax: plt.Axes, label: str = "c") -> None:
    df = prompt_panel_data()
    x = np.arange(len(DATASET_ORDER))
    width = 0.34
    matrix = 100 * df["min_matrix_agreement"].astype(float).to_numpy()
    prediction = 100 * df["min_prediction_agreement"].astype(float).to_numpy()
    bars_matrix = ax.bar(
        x - width / 2,
        matrix,
        width=width,
        color=PALETTE["blue"],
        edgecolor="white",
        linewidth=0.35,
        label="feature matrix",
        zorder=3,
    )
    bars_prediction = ax.bar(
        x + width / 2,
        prediction,
        width=width,
        color=PALETTE["gold"],
        edgecolor="white",
        linewidth=0.35,
        label="prediction",
        zorder=3,
    )
    annotate_vertical_bars(ax, bars_matrix, fmt=lambda v: f"{v:.1f}", dy=0.07, fontsize=5.0)
    annotate_vertical_bars(ax, bars_prediction, fmt=lambda v: f"{v:.1f}", dy=0.07, fontsize=5.0)
    ax.axhline(95, color="#9CA3AF", linewidth=0.65, linestyle="--", zorder=1)
    ax.text(
        len(DATASET_ORDER) - 0.54,
        95.10,
        "95%",
        fontsize=5.6,
        color=PALETTE["muted"],
        ha="right",
        va="bottom",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_DATASET[d] for d in DATASET_ORDER])
    ax.set_ylim(94, 100.35)
    ax.set_ylabel("Minimum agreement (%)")
    ax.set_title("Prompt stability")
    ax.legend(
        loc="upper right",
        frameon=False,
        ncol=1,
        borderaxespad=0.15,
        handletextpad=0.35,
        labelspacing=0.25,
    )
    clean_axis(ax)
    panel_label(ax, label)


def plot_audit_reliability(ax: plt.Axes) -> None:
    df = audit_panel_data()
    y_pos = {dataset: len(DATASET_ORDER) - 1 - idx for idx, dataset in enumerate(DATASET_ORDER)}
    y = np.array([y_pos[str(dataset)] for dataset in df["Dataset"]])
    ax.scatter(
        df["Primary human vs ASB kappa"],
        y + 0.10,
        s=30,
        color=PALETTE["teal"],
        edgecolor="white",
        linewidth=0.4,
        label="ASB vs human",
        zorder=3,
    )
    ax.scatter(
        df["Human-human kappa"],
        y - 0.10,
        s=30,
        color=PALETTE["purple"],
        edgecolor="white",
        linewidth=0.4,
        label="human-human",
        zorder=3,
    )
    for _, row in df.iterrows():
        yi = y_pos[str(row["Dataset"])]
        ax.plot(
            [row["Primary human vs ASB kappa"], row["Human-human kappa"]],
            [yi + 0.10, yi - 0.10],
            color=PALETTE["grid"],
            linewidth=0.8,
            zorder=1,
        )
    ax.set_yticks([y_pos[d] for d in DATASET_ORDER])
    ax.set_yticklabels([SHORT_DATASET[d] for d in DATASET_ORDER])
    ax.set_ylim(-0.45, len(DATASET_ORDER) - 0.55)
    ax.set_xlim(0.54, 0.86)
    ax.set_xlabel("Cohen's kappa")
    ax.set_title("Concept audit reliability")
    ax.legend(loc="lower right", frameon=False, borderaxespad=0.15, handletextpad=0.35)
    clean_axis(ax, grid_axis="x")
    panel_label(ax, "b")


def annotate_vertical_bars(
    ax: plt.Axes,
    bars,
    *,
    fmt,
    dy: float,
    fontsize: float = 4.9,
    rotation: int = 0,
) -> None:
    for bar in bars:
        value = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + dy,
            fmt(value),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=rotation,
            color=PALETTE["ink"],
            clip_on=False,
        )


def plot_main_performance_bars(ax: plt.Axes) -> None:
    df = performance_panel_data()
    x = np.arange(len(DATASET_ORDER))
    methods = list(METHOD_COLORS)
    offsets = np.linspace(-0.27, 0.27, len(methods))
    bar_width = 0.17
    short_names = {
        "TF-IDF + LR": "TF-IDF",
        "RoBERTa embedding + LR": "RoBERTa",
        "DistilBERT fine-tuning": "DistilBERT-FT",
        "ASB-LR (ours)": "ASB-LR",
    }
    for idx, method in enumerate(methods):
        values = []
        for dataset in DATASET_ORDER:
            match = df[(df["Dataset"] == dataset) & (df["Method"] == method)]
            values.append(100 * float(match["Macro-F1 numeric"].iloc[0]))
        is_asb = method == "ASB-LR (ours)"
        bars = ax.bar(
            x + offsets[idx],
            values,
            width=bar_width,
            color=METHOD_COLORS[method],
            alpha=1.0 if is_asb else 0.72,
            edgecolor=PALETTE["ink"] if is_asb else "white",
            linewidth=0.55 if is_asb else 0.35,
            label=short_names[method],
            zorder=3,
        )
        if is_asb:
            annotate_vertical_bars(
                ax,
                bars,
                fmt=lambda v: f"{v:.1f}",
                dy=0.25,
                fontsize=5.4,
                rotation=0,
            )
    ax.axhline(90, color="#9CA3AF", linewidth=0.55, linestyle="--", zorder=1)
    ax.text(
        len(DATASET_ORDER) - 0.58,
        90.25,
        "90%",
        fontsize=5.5,
        color=PALETTE["muted"],
        ha="right",
        va="bottom",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_DATASET[d] for d in DATASET_ORDER])
    ax.set_ylim(70, 101.2)
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("Predictive performance")
    ax.legend(
        loc="upper right",
        frameon=False,
        ncol=2,
        borderaxespad=0.15,
        handletextpad=0.35,
        columnspacing=0.75,
        labelspacing=0.28,
    )
    clean_axis(ax)
    panel_label(ax, "a", x=-0.16)


def plot_main_audit_bars(ax: plt.Axes) -> None:
    df = audit_panel_data()
    x = np.arange(len(DATASET_ORDER))
    width = 0.34
    asb = 100 * df["Primary human vs ASB kappa"].astype(float).to_numpy()
    human = 100 * df["Human-human kappa"].astype(float).to_numpy()
    bars_asb = ax.bar(
        x - width / 2,
        asb,
        width=width,
        color=PALETTE["teal"],
        edgecolor="white",
        linewidth=0.35,
        label="ASB-human",
        zorder=3,
    )
    bars_human = ax.bar(
        x + width / 2,
        human,
        width=width,
        color=PALETTE["purple"],
        edgecolor="white",
        linewidth=0.35,
        label="human-human",
        zorder=3,
    )
    annotate_vertical_bars(ax, bars_asb, fmt=lambda v: f"{v:.1f}", dy=0.55, fontsize=4.9)
    annotate_vertical_bars(ax, bars_human, fmt=lambda v: f"{v:.1f}", dy=0.55, fontsize=4.9)
    ax.axhline(60, color="#9CA3AF", linewidth=0.55, linestyle="--", zorder=1)
    ax.axhline(80, color="#9CA3AF", linewidth=0.55, linestyle=":", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_DATASET[d] for d in DATASET_ORDER])
    ax.set_ylim(50, 90)
    ax.set_ylabel("Cohen's kappa x100")
    ax.set_title("Concept audit reliability")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, -0.18),
        frameon=False,
        ncol=2,
        borderaxespad=0.15,
        handletextpad=0.35,
        columnspacing=0.75,
    )
    clean_axis(ax)
    panel_label(ax, "b", x=-0.16)


def plot_main_stability_bars(ax: plt.Axes) -> None:
    df = prompt_panel_data()
    x = np.arange(len(DATASET_ORDER))
    width = 0.34
    matrix = 100 * df["min_matrix_agreement"].astype(float).to_numpy()
    prediction = 100 * df["min_prediction_agreement"].astype(float).to_numpy()
    bars_matrix = ax.bar(
        x - width / 2,
        matrix,
        width=width,
        color=PALETTE["blue"],
        edgecolor="white",
        linewidth=0.35,
        label="feature matrix",
        zorder=3,
    )
    bars_prediction = ax.bar(
        x + width / 2,
        prediction,
        width=width,
        color=PALETTE["gold"],
        edgecolor="white",
        linewidth=0.35,
        label="prediction",
        zorder=3,
    )
    annotate_vertical_bars(ax, bars_matrix, fmt=lambda v: f"{v:.1f}", dy=0.08, fontsize=4.9)
    annotate_vertical_bars(ax, bars_prediction, fmt=lambda v: f"{v:.1f}", dy=0.08, fontsize=4.9)
    ax.axhline(95, color="#9CA3AF", linewidth=0.65, linestyle="--", zorder=1)
    ax.text(
        len(DATASET_ORDER) - 0.52,
        95.12,
        "95%",
        fontsize=5.5,
        color=PALETTE["muted"],
        ha="right",
        va="bottom",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_DATASET[d] for d in DATASET_ORDER])
    ax.set_ylim(94, 100.4)
    ax.set_ylabel("Minimum agreement (%)")
    ax.set_title("Prompt stability")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, -0.18),
        frameon=False,
        ncol=2,
        borderaxespad=0.15,
        handletextpad=0.35,
        columnspacing=0.75,
    )
    clean_axis(ax)
    panel_label(ax, "c", x=-0.16)


def plot_main_cost_performance(ax: plt.Axes) -> None:
    perf = provider_panel_data().copy()
    cost = cost_panel_data().copy()
    provider_map = {
        "DeepSeek V4 Flash": "DeepSeek",
        "GPT-4.1 mini": "OpenAI",
        "Qwen Plus": "Qwen",
        "Gemini 2.5 Flash": "Google",
    }
    short_provider = {
        "DeepSeek V4 Flash": "DeepSeek",
        "GPT-4.1 mini": "GPT-4.1",
        "Qwen Plus": "Qwen",
        "Gemini 2.5 Flash": "Gemini",
    }
    perf["Platform"] = perf["Feature extractor"].map(provider_map)
    perf["Provider"] = perf["Feature extractor"].map(short_provider)
    merged = perf.merge(cost[["Platform", "Estimated cost"]], on="Platform", how="left")
    write_source_data("main_claims_cost_performance", merged)

    dataset = "Disaster Tweets"
    plot_df = (
        merged[merged["Dataset"] == dataset]
        .sort_values("Estimated cost")
        .reset_index(drop=True)
    )
    ax.plot(
        plot_df["Estimated cost"],
        plot_df["Macro-F1"],
        color=PALETTE["blue"],
        linewidth=1.3,
        marker="o",
        markersize=4.4,
        markeredgecolor="white",
        markeredgewidth=0.45,
        zorder=3,
    )
    label_offsets = {
        "DeepSeek": (1.03, -0.0022, "left"),
        "Qwen": (1.07, 0.0025, "left"),
        "GPT-4.1": (1.07, 0.0020, "left"),
        "Gemini": (1.05, 0.0020, "left"),
    }
    for _, row in plot_df.iterrows():
        xmult, yoff, align = label_offsets[str(row["Provider"])]
        ax.text(
            float(row["Estimated cost"]) * xmult,
            float(row["Macro-F1"]) + yoff,
            str(row["Provider"]),
            fontsize=5.9,
            color=PALETTE["muted"],
            ha=align,
            va="bottom",
        )
    ax.set_xscale("log")
    ax.set_xlim(1.0, 45)
    ax.set_ylim(0.795, 0.818)
    ax.set_xlabel("Cost for 10,000 examples (USD)")
    ax.set_ylabel("ASB-LR Macro-F1")
    ax.set_title("Disaster Tweets cost-performance")
    clean_axis(ax)
    panel_label(ax, "d")


def plot_mi_selection(ax: plt.Axes, label: str = "c") -> None:
    df = mi_panel_data()
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["orange"], PALETTE["purple"]]
    for dataset, color in zip(DATASET_ORDER, colors):
        part = df[df["Dataset"] == dataset]
        ax.plot(
            part["#Features"],
            part["Delta Macro-F1 vs all"],
            marker="o",
            markersize=3.2,
            linewidth=1.2,
            color=color,
            label=SHORT_DATASET[dataset],
            zorder=3,
        )
    ax.axhline(0, color=PALETTE["ink"], linewidth=0.55)
    ax.set_xticks([5, 10, 15, 20])
    ax.set_xlabel("MI-selected features")
    ax.set_ylabel(r"$\Delta$ Macro-F1 vs. all 20")
    ax.set_title("Train-only feature selection")
    ax.legend(loc="lower right", frameon=False)
    clean_axis(ax)
    panel_label(ax, label)


def plot_counterfactuals(ax: plt.Axes, label: str = "d") -> None:
    df = counterfactual_panel_data()
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["orange"], PALETTE["purple"]]
    for dataset, color in zip(DATASET_ORDER, colors):
        part = df[df["Dataset"] == dataset]
        ax.plot(
            part["Edit budget"],
            part["Coverage"],
            marker="o",
            markersize=3.2,
            linewidth=1.2,
            color=color,
            label=SHORT_DATASET[dataset],
            zorder=3,
        )
    ax.set_xticks([1, 2, 3])
    ax.set_ylim(0.45, 1.03)
    ax.set_xlabel("Edit budget")
    ax.set_ylabel("Error coverage")
    ax.set_title("Semantic edit diagnostic")
    ax.legend(loc="lower right", frameon=False)
    clean_axis(ax)
    panel_label(ax, label)


def plot_provider_comparison(ax: plt.Axes, label: str = "e") -> None:
    df = provider_panel_data()
    providers = ["DeepSeek V4 Flash", "GPT-4.1 mini", "Qwen Plus", "Gemini 2.5 Flash"]
    x = np.arange(len(providers))
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["red"], PALETTE["purple"]]
    markers = ["o", "s", "^", "D"]
    for dataset, color, marker in zip(DATASET_ORDER, colors, markers):
        part = df[df["Dataset"] == dataset].set_index("Feature extractor").loc[providers]
        ax.plot(
            x,
            part["Macro-F1"],
            marker=marker,
            markersize=3.4,
            linewidth=1.1,
            color=color,
            label=SHORT_DATASET[dataset],
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["DeepSeek", "GPT-4.1", "Qwen", "Gemini"])
    ax.set_ylim(0.70, 0.98)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Extractor sensitivity")
    ax.legend(
        loc="lower left",
        frameon=False,
        borderaxespad=0.15,
        handletextpad=0.35,
        ncol=2,
        columnspacing=0.65,
    )
    clean_axis(ax)
    panel_label(ax, label)


def plot_api_cost(ax: plt.Axes) -> None:
    df = cost_panel_data()
    order = ["DeepSeek", "Qwen", "OpenAI", "Google"]
    df = df.set_index("Platform").loc[order].reset_index()
    colors = [PALETTE["teal"], PALETTE["green"], PALETTE["blue"], PALETTE["gold"]]
    y = np.arange(len(df))
    ax.barh(
        y,
        df["Estimated cost"],
        color=colors,
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    for yi, cost in zip(y, df["Estimated cost"]):
        ax.text(cost * 1.05, yi, f"${cost:.2f}", va="center", ha="left", fontsize=6.2, color=PALETTE["ink"])
    ax.set_yticks(y)
    ax.set_yticklabels(df["Platform"])
    ax.set_xscale("log")
    ax.set_xlabel("Estimated cost, USD")
    ax.set_title("API cost for 10,000 feature rows")
    clean_axis(ax, grid_axis="x")
    panel_label(ax, "f")


def plot_cost_performance_scatter() -> None:
    """Appendix figure: provider cost versus ASB-LR Macro-F1."""
    perf = provider_panel_data().copy()
    cost = cost_panel_data().copy()
    provider_map = {
        "DeepSeek V4 Flash": "DeepSeek",
        "GPT-4.1 mini": "OpenAI",
        "Qwen Plus": "Qwen",
        "Gemini 2.5 Flash": "Google",
    }
    short_provider = {
        "DeepSeek V4 Flash": "DeepSeek",
        "GPT-4.1 mini": "GPT-4.1",
        "Qwen Plus": "Qwen",
        "Gemini 2.5 Flash": "Gemini",
    }
    perf["Platform"] = perf["Feature extractor"].map(provider_map)
    perf["Provider"] = perf["Feature extractor"].map(short_provider)
    merged = perf.merge(cost[["Platform", "Estimated cost"]], on="Platform", how="left")
    write_source_data("appendix_cost_performance_scatter", merged)

    fig, ax = plt.subplots(figsize=(3.7, 2.7))
    colors = {"ADE": PALETTE["red"], "Disaster Tweets": PALETTE["blue"]}
    markers = {"ADE": "o", "Disaster Tweets": "s"}
    for dataset in ["ADE", "Disaster Tweets"]:
        part = merged[merged["Dataset"] == dataset]
        ax.scatter(
            part["Estimated cost"],
            part["Macro-F1"],
            s=34,
            marker=markers[dataset],
            color=colors[dataset],
            edgecolor="white",
            linewidth=0.45,
            label=dataset,
            zorder=3,
        )
    label_rows = merged.drop_duplicates("Provider").sort_values("Estimated cost")
    label_offsets = {
        "DeepSeek": (1.05, -0.0020, "left"),
        "Qwen": (1.05, 0.0032, "left"),
        "GPT-4.1": (1.06, 0.0022, "left"),
        "Gemini": (1.06, 0.0020, "left"),
    }
    for _, row in label_rows.iterrows():
        y = float(merged.loc[merged["Provider"] == row["Provider"], "Macro-F1"].max())
        xmult, yoff, align = label_offsets[str(row["Provider"])]
        ax.text(
            float(row["Estimated cost"]) * xmult,
            y + yoff,
            str(row["Provider"]),
            fontsize=6.2,
            color=PALETTE["muted"],
            ha=align,
            va="bottom",
        )
    ax.set_xscale("log")
    ax.set_xlim(1.0, 45)
    ax.set_ylim(0.72, 0.825)
    ax.set_xlabel("Estimated API cost for 10,000 examples (USD, log scale)")
    ax.set_ylabel("ASB-LR Macro-F1")
    ax.set_title("Cost-performance tradeoff by extractor")
    ax.legend(loc="lower right", frameon=False, borderaxespad=0.15, handletextpad=0.35)
    clean_axis(ax)
    save_figure(fig, "appendix_cost_performance_scatter")


def plot_group_utility(ax: plt.Axes) -> None:
    df = group_panel_data()
    y = np.arange(len(DATASET_ORDER))
    for yi, dataset in enumerate(DATASET_ORDER):
        part = df[df["Dataset"] == dataset].set_index("Feature set")
        all_value = float(part.loc["All features", "Macro-F1"])
        best_value = float(part.loc["Best single group", "Macro-F1"])
        group = str(part.loc["Best single group", "Feature group"])
        ax.plot(
            [best_value, all_value],
            [yi, yi],
            color=PALETTE["grid"],
            linewidth=1.0,
            zorder=1,
        )
        ax.scatter(
            best_value,
            yi,
            s=34,
            color=PALETTE["gold"],
            edgecolor="white",
            linewidth=0.4,
            label="best single group" if yi == 0 else None,
            zorder=3,
        )
        ax.scatter(
            all_value,
            yi,
            s=34,
            color=PALETTE["teal"],
            edgecolor="white",
            linewidth=0.4,
            label="all 20 features" if yi == 0 else None,
            zorder=3,
        )
        ax.text(
            min(best_value, all_value) - 0.008,
            yi + 0.20,
            fill(group, width=20),
            ha="left",
            va="bottom",
            fontsize=5.7,
            color=PALETTE["muted"],
        )
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT_DATASET[d] for d in DATASET_ORDER])
    ax.set_xlim(0.68, 0.97)
    ax.set_xlabel("Macro-F1")
    ax.set_title("Feature-group utility")
    ax.legend(loc="lower right", frameon=False)
    clean_axis(ax, grid_axis="x")
    panel_label(ax, "f")


def plot_results_summary() -> None:
    """Main-text visual summary for feature-budget and intervention diagnostics."""
    fig = plt.figure(figsize=(7.2, 2.35))
    gs = fig.add_gridspec(1, 2, wspace=0.34)
    plot_mi_selection(fig.add_subplot(gs[0, 0]), label="a")
    plot_counterfactuals(fig.add_subplot(gs[0, 1]), label="b")
    save_figure(fig, "asb_results_summary")


def plot_rq3_stability_sensitivity() -> None:
    """RQ3 figure: prompt stability and extractor sensitivity."""
    fig = plt.figure(figsize=(7.2, 2.65))
    gs = fig.add_gridspec(1, 2, wspace=0.38)
    plot_prompt_stability(fig.add_subplot(gs[0, 0]), label="a")
    plot_provider_comparison(fig.add_subplot(gs[0, 1]), label="b")
    save_figure(fig, "rq3_stability_sensitivity")


def plot_main_claims_summary() -> None:
    """Main-text compact Figure 2: performance, auditability, and stability."""
    fig = plt.figure(figsize=(7.2, 4.55))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.0], wspace=0.34, hspace=0.62)
    plot_main_performance_bars(fig.add_subplot(gs[0, 0]))
    plot_audit_reliability(fig.add_subplot(gs[0, 1]))
    plot_prompt_stability_main(fig.add_subplot(gs[1, 0]), label="c")
    plot_main_cost_performance(fig.add_subplot(gs[1, 1]))
    save_figure(fig, "main_claims_summary")


def qa_exports() -> None:
    """Lightweight Python-side QA: confirm exported rasters are nonblank."""
    from PIL import Image, ImageStat

    for stem in [
        "workflow_overview",
        "asb_results_summary",
        "rq3_stability_sensitivity",
        "main_claims_summary",
        "appendix_cost_performance_scatter",
    ]:
        path = OUT_DIR / f"{stem}.png"
        image = Image.open(path).convert("L")
        stat = ImageStat.Stat(image)
        if stat.extrema[0][0] == stat.extrema[0][1]:
            raise RuntimeError(f"Blank image export detected: {path}")
        print(f"QA {path.name}: {image.size[0]}x{image.size[1]}, gray std={stat.stddev[0]:.2f}")


def main() -> None:
    set_style()
    plot_workflow()
    plot_results_summary()
    plot_rq3_stability_sensitivity()
    plot_main_claims_summary()
    plot_cost_performance_scatter()
    qa_exports()
    print(f"Wrote figures to {OUT_DIR}")
    print(f"Wrote figure source data to {SOURCE_DIR}")


if __name__ == "__main__":
    main()
