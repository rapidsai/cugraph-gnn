#!/usr/bin/env python3
"""
Utilities to slice model performance by node degree.

Intended use: given arrays of node degrees, ground-truth labels, and predictions, compute
per-bucket metrics (deciles plus extreme heads/tails) and optionally visualize the trends. This
helps surface whether a model underperforms on tail or head nodes in scale-free graphs.

Decile naming follows degree: higher decile numbers correspond to higher-degree (more popular)
items, so "Decile 10" is the top-degree slice. Plots/tables are ordered high->low degree (popular
to rare) for readability.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def evaluate_by_degree_bucket(
    degree: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int | None = None
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """
    Compute accuracy/F1 per degree bucket plus explicit head/tail slices.

    Args:
        degree: Node degrees as a 1D numpy array.
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        num_classes: Optional class count for deterministic confusion-matrix shapes.

    Returns:
        results_df: DataFrame with bucket, acc, f1, and count columns.
        confusions: Mapping from bucket name to confusion matrix.
    """
    # Use quantile bins so each decile has comparable support; allow duplicate edges to avoid errors
    # when degree distributions are flat.
    deciles = pd.qcut(degree, 10, labels=False, duplicates="drop")

    # Separate head slices make it easy to see if a model ignores the absolute largest hubs.
    thr_5 = np.percentile(degree, 95)
    thr_1 = np.percentile(degree, 99)
    # And tail slices reveal whether the model struggles on sparsely connected nodes.
    bot_5 = np.percentile(degree, 5)
    bot_1 = np.percentile(degree, 1)

    top_5_mask = degree >= thr_5
    top_1_mask = degree >= thr_1
    bot_5_mask = degree <= bot_5
    bot_1_mask = degree <= bot_1

    results: list[dict[str, float]] = []
    confusions: dict[str, np.ndarray] = {}

    def eval_bucket(mask: np.ndarray, name: str) -> None:
        """Compute metrics for a boolean mask; appends to outer results/printouts."""
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return

        bucket_true = y_true[idx]
        bucket_pred = y_pred[idx]

        acc = accuracy_score(bucket_true, bucket_pred)
        f1 = f1_score(bucket_true, bucket_pred, average="macro")

        results.append({"bucket": name, "acc": acc, "f1": f1, "count": len(idx)})
        confusions[name] = confusion_matrix(
            bucket_true,
            bucket_pred,
            labels=list(range(num_classes)) if num_classes is not None else None,
        )

        print(f"\n{name}:")
        print("Nodes:", len(idx))
        print("Accuracy:", acc)
        print("F1:", f1)

    for decile in range(10):
        mask = deciles == decile
        eval_bucket(mask, f"Decile {decile + 1}")  # Decile 10 == highest degree

    eval_bucket(bot_5_mask, "Bottom 5%")
    eval_bucket(bot_1_mask, "Bottom 1%")
    eval_bucket(top_5_mask, "Top 5%")
    eval_bucket(top_1_mask, "Top 1%")

    results_df = pd.DataFrame(results)

    # Preserve a human-friendly order (high degree left -> low degree right).
    order = (
        ["Top 1%", "Top 5%"]
        + [f"Decile {i}" for i in range(10, 0, -1)]  # Decile 10 highest -> Decile 1 lowest
        + ["Bottom 5%", "Bottom 1%"]
    )
    results_df["order"] = results_df["bucket"].apply(order.index)
    results_df = results_df.sort_values("order")

    return results_df, confusions


def plot_performance(results_df: pd.DataFrame) -> None:
    """
    Plot accuracy and macro-F1 versus degree buckets.

    The x-axis carries bucket labels; rotation is enabled for readability when many buckets are
    present (deciles + head/tail slices). Higher-numbered deciles correspond to higher degree
    (more popular items).
    """
    # Respect explicit ordering if provided so the chart flows high->low degree left to right.
    if "order" in results_df.columns:
        results_df = results_df.sort_values("order")

    plt.figure(figsize=(10, 4))
    plt.plot(results_df["bucket"], results_df["acc"], marker="o", label="Accuracy")
    plt.plot(results_df["bucket"], results_df["f1"], marker="o", label="F1 Macro")

    plt.title("Performance vs Degree (Popularity) Decile + Heads/Tails")
    plt.xlabel("Degree Bucket")
    plt.ylabel("Performance")
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    raise SystemExit(
        "This module expects `degree`, `y_true`, and `y_pred` numpy arrays. "
        "Call `evaluate_by_degree_bucket` then `plot_performance` with real data."
    )
