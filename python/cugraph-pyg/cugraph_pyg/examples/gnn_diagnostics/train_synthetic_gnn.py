#!/usr/bin/env python3
"""
Minimal smoke-test GNN training on synthetic data.

Purpose: validate that torch/torch_geometric installations (plus CUDA wiring) work end-to-end by
training a tiny 2-layer GCN on a random graph. Meant to be quick, not performant.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import erdos_renyi_graph
from torch_geometric.data import Data


@dataclass
class Config:
    """Container for the synthetic graph shape, model size, and runtime parameters."""
    num_nodes: int = 5_000
    num_edges: int = 20_000
    num_features: int = 16
    num_classes: int = 4
    hidden_dim: int = 32
    epochs: int = 3
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GCN(nn.Module):
    """Two-layer GCN used for the smoke test."""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


def make_synthetic_data(cfg: Config) -> Data:
    """Generate an Erdos-Renyi graph plus random features/labels for node classification."""
    # Keep edge probability small so the graph remains sparse for the smoke test.
    edge_index = erdos_renyi_graph(cfg.num_nodes, cfg.num_edges / (cfg.num_nodes**2))
    x = torch.randn(cfg.num_nodes, cfg.num_features)
    y = torch.randint(0, cfg.num_classes, (cfg.num_nodes,))
    data = Data(x=x, edge_index=edge_index, y=y)
    return data


def train(cfg: Config) -> None:
    """
    Train the synthetic GCN and print train/val accuracy.

    Uses an 80/20 split on nodes for a quick sanity check; metrics are illustrative only.
    """
    # Fixed seeds keep the quick-check metrics stable across runs.
    random.seed(0)
    torch.manual_seed(0)
    device = torch.device(cfg.device)

    data = make_synthetic_data(cfg).to(device)

    model = GCN(cfg.num_features, cfg.hidden_dim, cfg.num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    # Simple train/val split to get a quick sanity signal without extra loaders.
    idx = torch.randperm(cfg.num_nodes, device=device)
    split = int(cfg.num_nodes * 0.8)  # 80/20 keeps enough train data while leaving a small check set
    train_idx = idx[:split]
    val_idx = idx[split:]

    print(f"Using device: {device}")
    print(f"Data: {cfg.num_nodes} nodes, {cfg.num_edges} edges, classes={cfg.num_classes}")
    model.train()
    for epoch in range(1, cfg.epochs + 1):
        total_loss = 0.0
        opt.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[train_idx], data.y[train_idx])
        loss.backward()
        opt.step()
        total_loss += loss.item()
        print(f"Epoch {epoch:02d} - loss: {total_loss:.4f}")

    # Quick eval on all nodes
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits.argmax(dim=-1)
        train_acc = (preds[train_idx] == data.y[train_idx]).float().mean().item()
        val_acc = (preds[val_idx] == data.y[val_idx]).float().mean().item()
    print(f"Train acc: {train_acc:.3f} | Val acc: {val_acc:.3f}")


def parse_args() -> Config:
    """CLI wrapper returning a populated Config (defaults prefer quick runtimes)."""
    parser = argparse.ArgumentParser(description="Synthetic GNN smoke test")
    parser.add_argument("--num-nodes", type=int, default=Config.num_nodes)
    parser.add_argument("--num-edges", type=int, default=Config.num_edges)
    parser.add_argument("--num-features", type=int, default=Config.num_features)
    parser.add_argument("--num-classes", type=int, default=Config.num_classes)
    parser.add_argument("--hidden-dim", type=int, default=Config.hidden_dim)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string (default: cuda if available else cpu)",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(
        num_nodes=args.num_nodes,
        num_edges=args.num_edges,
        num_features=args.num_features,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
    )
    return cfg


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
