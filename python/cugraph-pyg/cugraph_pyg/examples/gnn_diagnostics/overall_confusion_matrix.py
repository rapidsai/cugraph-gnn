#!/usr/bin/env python3
"""
Simple helper to render a confusion matrix from logits and labels.

Intended for quick diagnostics in notebooks/scripts without pulling in heavier evaluation tooling.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix


def plot_overall_confusion_matrix(
    labels: torch.Tensor, model_outputs: torch.Tensor, num_classes: int | None = None
) -> None:
    """
    Plot the confusion matrix given labels and raw model outputs (logits).

    Args:
        labels: Ground-truth labels tensor.
        model_outputs: Model outputs (logits or probabilities), shape (N, num_classes).
        num_classes: Optional explicit class count to lock axis ordering.
    """
    y_true = labels.detach().cpu().numpy()
    # Argmax on raw logits to avoid needing a softmax; we only need class decisions.
    y_pred = model_outputs.argmax(dim=1).detach().cpu().numpy()

    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(num_classes)) if num_classes is not None else None
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap="viridis")
    plt.title("Overall Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    raise SystemExit(
        "Provide tensors for `labels` and `model_outputs` and call "
        "`plot_overall_confusion_matrix(labels, model_outputs)`."
    )
