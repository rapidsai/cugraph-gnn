#!/usr/bin/env python3
"""
End-to-end demonstration of the diagnostics utilities on a controlled "rapid drop then plateau"
training trajectory.

Motivation
----------
The script is meant to be a reproducible, single-file demo that surfaces the diagnostic plots we
use elsewhere in the project. The training loop is intentionally configured to converge quickly and
then flatten so that the curvature and error distribution tools have something interesting to show.

What it builds
--------------
- A synthetic node-classification dataset with heavy-tailed degrees and degree-dependent labels,
  framed as item popularity (e.g., popular/high-degree, rare/low-degree).
- Classes are stratified by difficulty: high-degree items are easiest, mid-degree moderate, and
  low-degree hardest to predict.
- A tiny MLP classifier trained with Adam.
- A forced learning-rate drop after a few hundred steps to trigger an early plateau.

What it logs / saves (under `artifacts/`)
-----------------------------------------
- `loss_curve.png`: loss over steps with the forced LR drop annotated.
- `hessian_curve.png`: top Hessian eigenvalue samples while training (if collected).
- `confusion_matrix.png`: overall confusion matrix at the end of training.
- `degree_performance.png`: accuracy/F1 by degree decile.

Usage example (from repo root)
------------------------------
    python python/cugraph-pyg/cugraph_pyg/examples/gnn_diagnostics/synthetic_diagnostics_demo.py \\
        --device cuda --epochs 1000 --hessian-sample-every 10 --hessian-max-samples 100 \\
        --plateau-step 50 --plateau-lr-scale 0.1
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent

from overall_confusion_matrix import plot_overall_confusion_matrix
from degree_decile_performance import evaluate_by_degree_bucket, plot_performance
from hessian_top_eigen import estimate_top_eigenvalue_vhp, plot_curvature
from sklearn.metrics import confusion_matrix


@dataclass
class Config:
    """
    Container for all hyperparameters and runtime knobs.

    The defaults are chosen so that the script:
    - Runs quickly on CPU or GPU.
    - Hits an early sharp loss drop, then plateaus after `plateau_step`.
    - Collects a small number of Hessian samples for the curvature plot.
    """
    num_nodes: int = 8_000
    num_features: int = 16
    num_classes: int = 3
    batch_size: int = 512
    epochs: int = 3
    lr: float = 5e-3
    plateau_step: int = 400  # reduce the LR after this many steps to force plateau
    plateau_lr_scale: float = 0.1  # multiplier applied at plateau_step (set 0 for hard freeze)
    hessian_sample_every: int = 50
    hessian_max_samples: int = 30
    hessian_iters: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0


class MLP(nn.Module):
    """Minimal 2-layer MLP used purely for the synthetic classification task."""
    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_synthetic(cfg: Config) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Build a toy node-classification dataset with skewed degrees and aligned labels.

    Returns
    -------
    x_tensor : torch.Tensor
        Node features with the first dimension correlated with degree.
    y_tensor : torch.Tensor
        Integer class labels biased by degree bucket (head/mid/tail).
    degrees : np.ndarray
        Raw degree values used later for bucketed diagnostics.
    """
    rng = np.random.default_rng(cfg.seed)
    # Heavy-tailed degrees with fewer exact ties than Zipf to stabilize percentile cuts.
    raw_degrees = rng.lognormal(mean=1.0, sigma=0.75, size=cfg.num_nodes)
    degrees = raw_degrees.astype(np.float32)
    degrees = np.clip(degrees, a_min=1.0, a_max=None)  # avoid zero-degree edge cases

    # Inject degree signals with varying strength so classes differ in difficulty.
    x = rng.standard_normal((cfg.num_nodes, cfg.num_features)).astype(np.float32)
    degree_norm = degrees / degrees.max()

    # Bias labels by degree to mimic popularity: class 0 = popular items (high degree),
    # class 1 = mid-tier items, class 2 = rare/long-tail items (low degree).
    # Cutoffs create difficulty tiers: head (easy), mid (medium), tail (hard).
    p80 = np.percentile(degrees, 80)
    p40 = np.percentile(degrees, 40)
    p20 = np.percentile(degrees, 20)

    labels = np.ones(cfg.num_nodes, dtype=np.int64)
    head_mask = degrees >= p80
    mid_mask = (degrees >= p40) & (degrees < p80)
    tail_mask = degrees <= p20

    # Fallback to ensure every class is represented even if percentiles collapse.
    if not head_mask.any():
        head_idx = np.argsort(degrees)[-max(1, int(0.1 * cfg.num_nodes)) :]
        head_mask[head_idx] = True
    if not mid_mask.any():
        mid_idx = np.argsort(degrees)[int(0.4 * cfg.num_nodes) : int(0.8 * cfg.num_nodes)]
        mid_mask[mid_idx] = True
    if not tail_mask.any():
        tail_idx = np.argsort(degrees)[: max(1, int(0.1 * cfg.num_nodes))]
        tail_mask[tail_idx] = True

    labels[head_mask] = 0  # popular/high degree (easy)
    labels[mid_mask] = 1  # mid-tier/medium degree (medium)
    labels[tail_mask] = 2  # rare/low degree (hard)

    # Feature difficulty gradient:
    # - Head: strong, clean signal -> easy.
    x[:, 0] += degree_norm * 10.0 * head_mask
    # - Mid: moderate signal -> medium.
    x[:, 1] += degree_norm * 5.0 * mid_mask
    # - Tail: weak, noisy signal -> hard.
    tail_noise = rng.normal(0.0, 1.5, size=degree_norm.shape).astype(np.float32)
    x[:, 2] += (degree_norm * 2.0 + tail_noise) * tail_mask

    x_tensor = torch.from_numpy(x)

    y_tensor = torch.from_numpy(labels)

    return x_tensor, y_tensor, degrees


def train(cfg: Config) -> None:
    """
    Train the synthetic task and emit diagnostics/artifacts.

    The loop:
    - Trains for `cfg.epochs` epochs or until ~`plateau_step + 600` steps.
    - Forces a learning-rate drop at `cfg.plateau_step` to create a visible plateau.
    - Samples Hessian top eigenvalues every `cfg.hessian_sample_every` steps (up to max samples).
    - Saves loss, curvature, confusion matrix, and degree-bucket performance plots to `artifacts/`.
    """
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    x, y, degrees = make_synthetic(cfg)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model = MLP(cfg.num_features, hidden=64, out_dim=cfg.num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    step = 0
    losses = []
    hess_steps = []
    hess_vals = []

    model.train()
    for _ in range(cfg.epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            if step == cfg.plateau_step:
                for g in opt.param_groups:
                    # Force a learning-rate drop mid-training to create a visible plateau.
                    g["lr"] = g["lr"] * cfg.plateau_lr_scale

            opt.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            opt.step()

            losses.append(loss.item())
            if step % cfg.hessian_sample_every == 0 and len(hess_steps) < cfg.hessian_max_samples:
                ev = estimate_top_eigenvalue_vhp(
                    model=model,
                    batch=(batch_x, batch_y),
                    loss_fn=criterion,
                    iterations=cfg.hessian_iters,
                )
                hess_steps.append(step)
                hess_vals.append(ev)
                # Log loss alongside curvature so spikes/plateaus can be correlated.
                print(f"[HESSIAN] step={step} loss={loss.item():.6g} ev={ev:.6g}")

            step += 1
            if step >= cfg.plateau_step + 600:  # cap total steps so the demo stays quick
                break
        else:
            continue
        break

    min_loss = min(losses)
    min_step = losses.index(min_loss)
    print(
        f"Trained for {step} steps; "
        f"loss start={losses[0]:.4f} min={min_loss:.4f}@{min_step} final={losses[-1]:.4f}"
    )
    out_dir = ROOT / "artifacts"
    out_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(6, 3))
    plt.plot(losses, label="Loss")
    plt.axvline(cfg.plateau_step, color="red", linestyle="--", label="LR zeroed")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Rapid drop then plateau")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=150)
    plt.close()

    if hess_steps:
        plot_curvature(hess_steps, hess_vals)
        plt.savefig(out_dir / "hessian_curve.png", dpi=150)
        plt.close()
        paired = ", ".join(f"{s}:{v:.6g}" for s, v in zip(hess_steps, hess_vals))
        print(f"\nRecorded {len(hess_steps)} Hessian samples (step:eigen): {paired}")
        print("Saved artifacts/hessian_curve.png")
    else:
        print("\nNo Hessian samples recorded.")

    # Full inference for diagnostics
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device)).cpu()
    y_pred = logits.argmax(dim=1)
    pred_counts = np.bincount(y_pred.numpy(), minlength=cfg.num_classes)
    true_counts = np.bincount(y.numpy(), minlength=cfg.num_classes)
    print(f"\nLabel distribution (true): {true_counts}")
    print(f"Label distribution (pred): {pred_counts}")

    # Overall confusion matrix
    print("\nOverall confusion matrix:")
    cm = confusion_matrix(y, y_pred, labels=list(range(cfg.num_classes)))
    print(cm)
    plot_overall_confusion_matrix(labels=y, model_outputs=logits, num_classes=cfg.num_classes)
    plt.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # Degree decile performance
    results_df, confusions = evaluate_by_degree_bucket(
        degrees.astype(np.float32), y.numpy(), y_pred.numpy(), num_classes=cfg.num_classes
    )
    print("\nDegree-bucket metrics:")
    print(results_df[["bucket", "acc", "f1", "count"]])
    plot_performance(results_df)
    plt.savefig(out_dir / "degree_performance.png", dpi=150)
    plt.close()

    # Hessian top eigenvalue on a final batch
    sample_batch = next(iter(loader))
    sample_batch = (sample_batch[0].to(device), sample_batch[1].to(device))
    ev = estimate_top_eigenvalue_vhp(
        model=model, batch=sample_batch, loss_fn=criterion, iterations=10
    )
    print(f"\nEstimated top Hessian eigenvalue (sample batch): {ev:.4f}")


def parse_args() -> Config:
    """
    Parse CLI flags into a `Config`.

    Note: `--device` defaults to CUDA when available; passing `--device cpu` forces CPU training.
    """
    parser = argparse.ArgumentParser(description="Synthetic diagnostics demo")
    parser.add_argument("--num-nodes", type=int, default=Config.num_nodes)
    parser.add_argument("--num-features", type=int, default=Config.num_features)
    parser.add_argument("--num-classes", type=int, default=Config.num_classes)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--plateau-step", type=int, default=Config.plateau_step)
    parser.add_argument("--plateau-lr-scale", type=float, default=Config.plateau_lr_scale)
    parser.add_argument("--hessian-sample-every", type=int, default=Config.hessian_sample_every)
    parser.add_argument("--hessian-max-samples", type=int, default=Config.hessian_max_samples)
    parser.add_argument("--hessian-iters", type=int, default=Config.hessian_iters)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return Config(
        num_nodes=args.num_nodes,
        num_features=args.num_features,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        plateau_step=args.plateau_step,
        plateau_lr_scale=args.plateau_lr_scale,
        hessian_sample_every=args.hessian_sample_every,
        hessian_max_samples=args.hessian_max_samples,
        hessian_iters=args.hessian_iters,
        device=device,
    )


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
