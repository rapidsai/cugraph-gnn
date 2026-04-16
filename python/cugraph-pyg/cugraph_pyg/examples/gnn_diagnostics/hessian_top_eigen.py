#!/usr/bin/env python3
"""
Support for estimating and visualizing loss curvature via top Hessian eigenvalues.

The main entrypoint `estimate_top_eigenvalue_vhp` uses PyTorch's vector-Jacobian products to perform
power-iteration on the Hessian. `train_with_hessian_sampling` shows how to integrate sampling into a
training loop; `plot_curvature` visualizes the resulting trajectory.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Tuple

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn, optim
from torch.autograd.functional import vhp

try:
    from torch.func import functional_call  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call  # type: ignore


def _flatten(params: Iterable[Tensor]) -> Tensor:
    """Flatten an iterable of tensors into a single 1D tensor."""
    return torch.cat([p.reshape(-1) for p in params])


def _unflatten(flat: Tensor, params: List[Tensor]) -> List[Tensor]:
    """Expand a flat tensor into a list of tensors shaped like `params`."""
    reshaped: List[Tensor] = []
    offset = 0

    for param in params:
        numel = param.numel()
        reshaped.append(flat[offset : offset + numel].view_as(param))
        offset += numel

    return reshaped


def estimate_top_eigenvalue_vhp(
    model: nn.Module,
    batch: Tuple[Tensor, Tensor],
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    iterations: int = 20,
) -> float:
    """
    Estimate the largest Hessian eigenvalue of the loss with respect to model parameters.

    Args:
        model: Neural network model (parameters must require gradients).
        batch: Tuple of (inputs, targets) tensors used to form the loss surface.
        loss_fn: Callable mapping (preds, labels) -> scalar Tensor.
        iterations: Power-iteration steps; more gives a tighter estimate but costs more compute.

    Returns:
        Estimated top Hessian eigenvalue as a Python float.
    """
    # Ignore frozen parameters so curvature computation only tracks learnable weights.
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    param_names = [n for n, _ in named_params]
    params: List[Tensor] = [p for _, p in named_params]

    X, y = batch

    def wrapped_loss(*params_tuple: Tensor) -> Tensor:
        """
        Compute loss with stateless functional call using provided params.

        Using `functional_call` keeps the model pure/side-effect free inside VHP.
        """
        param_dict = {name: p for name, p in zip(param_names, params_tuple)}
        preds = functional_call(model, param_dict, (X,))
        return loss_fn(preds, y)

    flat_params = _flatten(params)
    # Start power iteration from a random unit vector to avoid aligning with any single parameter.
    vec = torch.randn_like(flat_params)
    vec /= vec.norm()

    eigenvalue = torch.tensor(0.0, device=vec.device)

    for _ in range(iterations):
        # Re-shape the probe vector into parameter-shaped tensors for VHP.
        vec_as_params = _unflatten(vec, params)

        # vhp gives Hessian-vector product without materializing the full Hessian.
        _, hvp_tuple = vhp(wrapped_loss, tuple(params), tuple(vec_as_params), create_graph=False)
        hv = _flatten(list(hvp_tuple))

        # Power iteration update: project HV back onto the direction and renormalize.
        eigenvalue = torch.dot(vec, hv)
        hv_norm = hv.norm()
        if hv_norm == 0 or torch.isnan(hv_norm):
            return 0.0
        vec = hv / hv_norm

    return float(eigenvalue.item())


def train_with_hessian_sampling(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: Iterable[Tuple[Tensor, Tensor]],
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    sample_every: int = 20,
    max_samples: int = 30,
    hv_iters: int = 20,
) -> Tuple[List[int], List[float]]:
    """
    Train a model and periodically sample Hessian curvature.

    This is a convenience wrapper for demos and experiments; it assumes a standard supervised
    minibatch loop and stops once `max_samples` curvature points are collected.

    Args:
        model: Neural network.
        optimizer: PyTorch optimizer.
        train_loader: Iterable yielding (X, y) minibatches.
        loss_fn: Loss function.
        sample_every: Sample Hessian every N steps.
        max_samples: Stop after this many Hessian samples.
        hv_iters: Power iteration steps inside Hessian eigen computation.

    Returns:
        (steps, eigenvalues): Lists of step indices and Hessian eigenvalues.
    """
    curvature_steps: List[int] = []
    curvature_vals: List[float] = []

    device = next(model.parameters()).device
    step = 0
    collected = 0

    for _epoch in range(10_000):  # oversized sentinel; loop exits via collected count
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            model.train()
            optimizer.zero_grad()

            preds = model(X)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

            if step % sample_every == 0 and collected < max_samples:
                print(f"[Step {step}] Computing top Hessian eigenvalue...")

                eigen = estimate_top_eigenvalue_vhp(
                    model=model, batch=(X, y), loss_fn=loss_fn, iterations=hv_iters
                )

                # Store sparse samples only to keep the loop lightweight while still capturing trend.
                curvature_steps.append(step)
                curvature_vals.append(eigen)
                collected += 1

            step += 1

            if collected >= max_samples:
                return curvature_steps, curvature_vals

    return curvature_steps, curvature_vals


def plot_curvature(curvature_steps: list[int], curvature_vals: list[float]) -> None:
    """Plot curvature evolution across training steps."""
    plt.figure(figsize=(7, 4))
    plt.plot(curvature_steps, curvature_vals, marker="o")
    plt.xlabel("Training Step")
    plt.ylabel("Top Hessian Eigenvalue")
    plt.title("Curvature Evolution During Training")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    raise SystemExit(
        "Construct your model/optimizer/train_loader and call "
        "`train_with_hessian_sampling` followed by `plot_curvature`."
    )
