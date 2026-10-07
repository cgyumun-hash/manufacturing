"""Compact redesign of representative bar figures from main.

Data loading, preprocessing, and statistics are imported unchanged from the
main-branch scripts; only the drawing code differs. Outputs are written to
figures_refined/<stage>/ so the original figures/ folders stay untouched.

Run from anywhere:
    python figures_refined/render_refined_figures.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "manufacturing_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
NORMAL = "#2F6DB5"
ANOMALY = "#D62728"
COLORS = {"normal": NORMAL, "anomaly": ANOMALY}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


dq = load_module("analyze_data_quality", ROOT / "02_data_quality/scripts/analyze_data_quality.py")
bs = load_module("analyze_basic_statistics", ROOT / "03_basic_statistics/scripts/analyze_basic_statistics.py")


def apply_style() -> None:
    sns.set_theme(context="paper", style="ticks")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.color": "black",
            "axes.labelcolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "normal",
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 10,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "grid.linewidth": 0.4,
            "grid.color": "#DDDDDD",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, folder: str, name: str) -> None:
    target = OUT / folder
    target.mkdir(parents=True, exist_ok=True)
    fig.savefig(target / name, dpi=200, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def record_count_figure(segmented) -> None:
    """02_data_quality/01_record_counts.png: rows after exact-duplicate removal."""
    labels = ("normal", "anomaly")
    counts = [len(segmented[label]) for label in labels]
    fig, ax = plt.subplots(figsize=(2.6, 2.8))
    x = np.arange(len(labels))
    ax.bar(x, counts, width=0.45, color=[COLORS[label] for label in labels], alpha=0.85, zorder=2)
    for pos, value in zip(x, counts):
        ax.annotate(f"{value:,}", (pos, value), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [label.title() for label in labels])
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylim(0, max(counts) * 1.12)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[5, 10], integer=True))
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
    ax.set_ylabel("Rows")
    ax.set_title("Rows after duplicate removal", loc="left")
    ax.grid(axis="y", zorder=0)
    fig.tight_layout()
    save(fig, "02_data_quality", "01_record_counts.png")


def segment_lengths_figure(raw, segmented) -> None:
    """02_data_quality/03_segment_lengths.png: bar + line, normal vs anomaly panels."""
    eligibility = dq.create_tables(raw, segmented)["input_length_eligibility"]

    fig, axes = plt.subplots(2, 1, figsize=(5.6, 4.2), sharex=True)
    for ax, label in zip(axes, ("normal", "anomaly")):
        subset = eligibility[eligibility["class"] == label]
        x = subset.input_seconds.to_numpy()
        y = subset.eligible_segments.to_numpy()
        ax.bar(x, y, width=0.2, color=COLORS[label], alpha=0.85, zorder=2)
        ax.plot(x, y, color="#333333", marker="o", markersize=2.8, linewidth=0.9, zorder=3)
        for seconds, count in zip(x, y):
            ax.annotate(str(count), (seconds, count), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)
        ax.set_title(label.title(), loc="left")
        ax.set_ylabel("Eligible segments")
        ax.set_ylim(0, y.max() * 1.18)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        ax.grid(axis="y", zorder=0)
    axes[-1].set_xlabel("Input duration (seconds)")
    axes[-1].set_xticks(np.arange(0.5, 5.1, 0.5))
    fig.suptitle("Eligible segments by input duration (separate y-scales)", x=0.02, ha="left")
    fig.tight_layout(h_pad=0.8)
    save(fig, "02_data_quality", "03_segment_lengths.png")


def rms_shape_figure(stats) -> None:
    """03_basic_statistics/03_rms_shape_crest.png: 2x2 grouped bars."""
    metrics = [
        ("rms", "RMS", "RMS (log scale)"),
        ("skewness", "Skewness", "Skewness"),
        ("excess_kurtosis", "Excess kurtosis", "Excess kurtosis"),
        ("crest_factor", "Crest factor", "Crest factor"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6))
    x = np.arange(len(bs.SENSORS))
    width = 0.26
    gap = 0.03
    for ax, (metric, title, ylabel) in zip(axes.flat, metrics):
        for sign, label in ((-1, "normal"), (1, "anomaly")):
            values = stats[stats["class"] == label].set_index("sensor").loc[bs.SENSORS, metric]
            ax.bar(x + sign * (width + gap) / 2, values, width, label=label.title(),
                   color=COLORS[label], alpha=0.85, zorder=2)
        ax.set_xticks(x, [bs.SHORT_NAMES[s] for s in bs.SENSORS])
        ax.set_xlim(-0.6, len(x) - 0.4)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if metric == "rms":
            ax.set_yscale("log")
        else:
            ax.axhline(0, color="black", linewidth=0.6, zorder=3)
        ax.grid(axis="y", zorder=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.0),
               handlelength=1.0, columnspacing=1.0)
    fig.suptitle("Signal magnitude and shape statistics", x=0.02, ha="left")
    fig.tight_layout(h_pad=1.0, w_pad=1.2)
    save(fig, "03_basic_statistics", "03_rms_shape_crest.png")


def effect_size_figure(effect) -> None:
    """03_basic_statistics/08_effect_sizes.png: horizontal bars."""
    names = []
    for sensor in bs.SENSORS:
        names.extend([f"{bs.SHORT_NAMES[sensor]} raw", f"{bs.SHORT_NAMES[sensor]} |x|"])
    ordered = effect.copy()
    ordered["sensor_order"] = ordered.sensor.map({s: i for i, s in enumerate(bs.SENSORS)})
    ordered["repr_order"] = ordered.representation.map({"raw_signed": 0, "absolute_magnitude": 1})
    ordered = ordered.sort_values(["sensor_order", "repr_order"])
    y = np.arange(len(ordered))

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), sharey=True)
    for ax, metric, title in (
        (axes[0], "hedges_g", "Hedges' g"),
        (axes[1], "cliffs_delta", "Cliff's delta"),
    ):
        values = ordered[metric].to_numpy()
        ax.barh(y, values, height=0.5, color=np.where(values >= 0, ANOMALY, NORMAL), alpha=0.85, zorder=2)
        ax.axvline(0, color="black", linewidth=0.6, zorder=3)
        if metric == "cliffs_delta":
            for boundary in (-0.474, -0.33, -0.147, 0.147, 0.33, 0.474):
                ax.axvline(boundary, color="gray", linestyle=":", linewidth=0.6, zorder=1)
            ax.set_xlim(-1, 1)
        else:
            ax.set_xlim(min(values.min(), 0) - 0.6, values.max() * 1.18)
        for pos, value in zip(y, values):
            ax.annotate(f"{value:.2f}", (value, pos), xytext=(3 if value >= 0 else -3, 0),
                        textcoords="offset points", ha="left" if value >= 0 else "right",
                        va="center", fontsize=7, zorder=4,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2})
        ax.set_yticks(y, names)
        ax.set_title(title)
        ax.set_xlabel("Positive = anomaly larger")
        ax.grid(axis="x", zorder=0)
    axes[0].invert_yaxis()
    axes[1].tick_params(axis="y", length=0)
    fig.suptitle("Normal–anomaly effect sizes (all cleaned rows)", x=0.02, ha="left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "03_basic_statistics", "08_effect_sizes.png")


def main() -> None:
    apply_style()
    raw, cleaned = dq.load_sources()
    segmented = {label: dq.add_segments(frame) for label, frame in cleaned.items()}
    record_count_figure(segmented)
    segment_lengths_figure(raw, segmented)
    frames = bs.load_data()
    stats = bs.sensor_statistics(frames)
    effect = bs.effect_sizes(frames)
    rms_shape_figure(stats)
    effect_size_figure(effect)
    print(f"Refined figures written under: {OUT}")


if __name__ == "__main__":
    main()
