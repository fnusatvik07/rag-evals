"""Visualization utilities for evaluation results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_metric_bars(
    results_df: pd.DataFrame,
    metric_cols: list[str],
    title: str = "Average Score by Metric",
    threshold: float = 0.7,
    save_path: str | None = None,
) -> plt.Figure:
    """Horizontal bar chart of average metric scores."""
    avg = results_df[metric_cols].mean().sort_values(ascending=True)
    colors = [
        "#4CAF50" if s >= threshold else "#FF9800" if s >= 0.5 else "#F44336"
        for s in avg
    ]

    fig, ax = plt.subplots(figsize=(10, max(4, len(avg) * 0.5)))
    ax.barh(range(len(avg)), avg.values, color=colors)
    ax.set_yticks(range(len(avg)))
    ax.set_yticklabels(avg.index, fontsize=9)
    ax.set_xlabel("Average Score")
    ax.set_title(title)
    ax.axvline(x=threshold, color="gray", linestyle="--", alpha=0.5, label=f"Threshold ({threshold})")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    return fig


def plot_metric_heatmap(
    results_df: pd.DataFrame,
    metric_cols: list[str],
    title: str = "Metric Correlation Heatmap",
    save_path: str | None = None,
) -> plt.Figure:
    """Correlation heatmap of metrics."""
    corr = results_df[metric_cols].corr()

    fig, ax = plt.subplots(figsize=(max(8, len(metric_cols)), max(6, len(metric_cols) * 0.8)))
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)

    ax.set_xticks(range(len(metric_cols)))
    ax.set_yticks(range(len(metric_cols)))
    ax.set_xticklabels(metric_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(metric_cols, fontsize=8)

    for i in range(len(metric_cols)):
        for j in range(len(metric_cols)):
            val = corr.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im)
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    return fig


def plot_score_distribution(
    results_df: pd.DataFrame,
    metric_cols: list[str],
    title: str = "Score Distributions",
    save_path: str | None = None,
) -> plt.Figure:
    """Box plots of metric score distributions."""
    fig, ax = plt.subplots(figsize=(max(8, len(metric_cols) * 1.2), 6))
    results_df[metric_cols].boxplot(ax=ax, rot=45)
    ax.set_title(title)
    ax.set_ylabel("Score")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    return fig


def plot_comparison(
    comparison_data: list[dict],
    metric_cols: list[str],
    title: str = "Configuration Comparison",
    save_path: str | None = None,
) -> plt.Figure:
    """Grouped bar chart comparing configurations."""
    df = pd.DataFrame(comparison_data)
    configs = df["config"].tolist() if "config" in df else [f"Config {i}" for i in range(len(df))]

    x = np.arange(len(metric_cols))
    width = 0.8 / len(configs)

    fig, ax = plt.subplots(figsize=(max(10, len(metric_cols) * 2), 6))

    for i, (_, row) in enumerate(df.iterrows()):
        values = [row.get(m, 0) for m in metric_cols]
        ax.bar(x + i * width, values, width, label=configs[i])

    ax.set_xticks(x + width * (len(configs) - 1) / 2)
    ax.set_xticklabels(metric_cols, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
    return fig
