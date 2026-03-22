"""
Plot benchmark summaries from llmSummary.csv (v2 schema or legacy v1).

Writes figures under ./histograms/ — bar comparison by model, run distributions, optional KDE.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark_utils import normalize_dataframe_columns

try:
    import seaborn as sns

    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

OUTPUT_DIR = "histograms"
# Bump when output filenames or layout change (documented in LATEST_*.txt).
HISTOGRAM_SCHEMA_VERSION = "v2"

# Metrics to visualize (canonical v2 names)
METRICS: list[tuple[str, str]] = [
    ("latency_mean_s", "Mean latency (s)"),
    ("latency_p95_s", "P95 latency (s)"),
    ("latency_std_s", "Latency std (s)"),
    ("memory_mean_mb", "Mean RSS (MB)"),
    ("tokens_per_sec_mean", "Throughput (tok/s est.)"),
    ("num_parameters", "Parameter count"),
]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:80]


def _setup_style() -> None:
    if _HAS_SNS:
        sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)
    else:
        for candidate in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
            if candidate in plt.style.available:
                plt.style.use(candidate)
                break
        else:
            plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "#fafafa",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#cccccc",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "figure.titlesize": 13,
            "xtick.labelsize": 8,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "font.family": "sans-serif",
        }
    )


def _short_label(name: str, max_len: int = 28) -> str:
    name = str(name)
    return name if len(name) <= max_len else name[: max_len - 1] + "…"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _save_figure(fig, basename: str, stamp: str, dpi: int, written: list[str]) -> None:
    """Write stable path histograms/<name> (latest) and archive histograms/versions/<stamp>/<name>."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stable = os.path.join(OUTPUT_DIR, basename)
    fig.savefig(stable, dpi=dpi, bbox_inches="tight")
    vdir = os.path.join(OUTPUT_DIR, "versions", stamp)
    os.makedirs(vdir, exist_ok=True)
    fig.savefig(os.path.join(vdir, basename), dpi=dpi, bbox_inches="tight")
    written.append(basename)


def _write_manifest(stamp: str, written: list[str]) -> None:
    path = os.path.join(OUTPUT_DIR, f"LATEST_{HISTOGRAM_SCHEMA_VERSION}.txt")
    lines = [
        f"schema={HISTOGRAM_SCHEMA_VERSION}",
        f"stamp_utc={stamp}",
        f"generated_utc={datetime.now(timezone.utc).isoformat()}",
        "",
        "Stable paths (always overwritten on each visualize run):",
    ]
    for name in sorted(set(written)):
        lines.append(f"  histograms/{name}")
    lines.extend(["", f"Versioned copy of this run: histograms/versions/{stamp}/"])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def visualize_data(filename: str = "llmSummary.csv") -> None:
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Missing {filename}; run ModelRunner first.")

    df = pd.read_csv(filename)
    if df.empty:
        raise ValueError(f"{filename} is empty.")

    df = normalize_dataframe_columns(df)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _setup_style()

    stamp = _utc_stamp()
    written: list[str] = []

    _plot_run_distributions(df, stamp, written)
    _plot_model_comparison(df, stamp, written)
    _write_manifest(stamp, written)
    _maybe_show()


def _maybe_show() -> None:
    if os.environ.get("MPL_SHOW", "").lower() in ("1", "true", "yes"):
        plt.show()
    else:
        plt.close("all")


def _plot_run_distributions(df: pd.DataFrame, stamp: str, written: list[str]) -> None:
    """Histogram / KDE of each metric across runs (one row = one run)."""
    for col, title in METRICS:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        vals = series.values
        ax.hist(vals, bins=min(30, max(8, len(vals) // 2)), color="#2563eb", alpha=0.75, edgecolor="white")
        if _HAS_SNS and len(vals) >= 4:
            try:
                sns.kdeplot(vals, ax=ax, color="#dc2626", linewidth=1.5, warn_singular=False)
            except Exception:
                pass
        ax.set_title(f"Distribution — {title}")
        ax.set_xlabel(title)
        ax.set_ylabel("Runs")
        fig.tight_layout()
        _save_figure(fig, f"{_slug(col)}_distribution.png", stamp, 160, written)
        plt.close(fig)


def _plot_model_comparison(df: pd.DataFrame, stamp: str, written: list[str]) -> None:
    """Bar charts: per model, mean metric ± std across runs."""
    if "model_name" not in df.columns:
        return

    if df["model_name"].dropna().empty:
        return

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 11))
    axes_flat: Iterable = axes.flatten()

    metric_cols = [m for m, _ in METRICS if m in df.columns][:6]
    for ax, col in zip(axes_flat, metric_cols):
        sub = df[["model_name", col]].copy()
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.dropna(subset=[col])
        if sub.empty:
            ax.set_visible(False)
            continue

        agg = sub.groupby("model_name")[col].agg(["mean", "std", "count"]).reset_index()
        agg["std"] = agg["std"].fillna(0.0)
        ascending = col.startswith("latency")
        agg = agg.sort_values("mean", ascending=ascending).reset_index(drop=True)

        labels = [_short_label(m) for m in agg["model_name"]]
        x = np.arange(len(agg))
        means = agg["mean"].values
        errs = agg["std"].values
        errs = np.where(agg["count"].values > 1, errs, 0)

        ax.bar(x, means, yerr=errs, capsize=3, color="#0d9488", alpha=0.85, ecolor="#444")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=42, ha="right")
        ylab = next((t for c, t in METRICS if c == col), col)
        ax.set_ylabel(ylab)
        ax.set_title(f"By model — {ylab}")

    for ax in axes_flat[len(metric_cols) :]:
        ax.set_visible(False)

    fig.suptitle("Benchmark comparison by model (mean ± std across runs)", fontsize=13, y=1.02)
    fig.tight_layout()
    _save_figure(fig, "summary_by_model.png", stamp, 170, written)
    plt.close(fig)

    # Legacy-style overlay histograms (per metric, colored by model) when few points per model
    _legacy_overlay_histograms(df, stamp, written)


def _legacy_overlay_histograms(df: pd.DataFrame, stamp: str, written: list[str]) -> None:
    """When multiple runs exist per model, show overlapping normalized histograms."""
    if "model_name" not in df.columns:
        return

    models = df["model_name"].dropna().unique().tolist()
    if len(models) < 2:
        return

    cmap = plt.get_cmap("tab10")
    for col, title in METRICS:
        if col not in df.columns:
            continue
        sub = df[["model_name", col]].copy()
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.dropna()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        vmin, vmax = sub[col].min(), sub[col].max()
        if vmin == vmax:
            vmax = vmin + 1e-9
        bins = 12

        for i, model in enumerate(models):
            pts = sub.loc[sub["model_name"] == model, col].values
            if len(pts) < 2:
                continue
            ax.hist(
                pts,
                bins=bins,
                range=(vmin, vmax),
                alpha=0.45,
                label=_short_label(model, 36),
                color=cmap(i % 10),
                density=True,
            )

        ax.set_title(f"Overlapping distributions — {title}")
        ax.set_xlabel(title)
        ax.set_ylabel("Density")
        _handles, leg_labels = ax.get_legend_handles_labels()
        if leg_labels:
            ax.legend(loc="best", fontsize=7)
        fig.tight_layout()
        _save_figure(fig, f"{_slug(col)}_by_model_hist.png", stamp, 150, written)
        plt.close(fig)


if __name__ == "__main__":
    visualize_data()
