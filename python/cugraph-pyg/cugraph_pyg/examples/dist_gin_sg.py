# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import time
import argparse
import torch
import torch.nn.functional as F

from torch_geometric.nn import MLP, GINConv, global_add_pool
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import OneHotDegree
from torch.utils.data import Dataset, DataLoader

from cugraph_pyg.data import FeatureStore
from cugraph_pyg.tensor import DistTensor
import torch.distributed as dist


class DistTensorGraphDataset(Dataset):
    """Optimized dataset for extracting individual graphs from distributed tensors."""

    def __init__(
        self, dist_edge_index, feature_store, device, edge_ptr, graph_indices=None
    ):
        self.dist_edge_index = dist_edge_index
        self.feature_store = feature_store
        self.device = device

        # Store edge pointer for graph access
        self._edge_ptr = edge_ptr

        if graph_indices is not None:
            self.graph_indices = graph_indices
        else:
            # Get num_graphs from edge_ptr length
            num_graphs = len(edge_ptr) - 1
            self.graph_indices = list(range(num_graphs))

        # Cache graph labels to avoid repeated feature store access
        self._cached_labels = None
        if len(self.graph_indices) < 1000:  # Only cache for smaller datasets
            self._cached_labels = self.feature_store["graph", "y", None][
                torch.tensor(self.graph_indices, device=self.device)
            ]

    def __len__(self):
        return len(self.graph_indices)

    def __getitem__(self, idx):
        # Get graph label - use cached version if available
        if self._cached_labels is not None:
            y = self._cached_labels[idx]
        else:
            y = self.feature_store["graph", "y", None][
                torch.tensor([idx], device=self.device)
            ]
        if y.dim() > 1:
            y = y.squeeze()

        # Get edges for this graph using pre-computed pointers
        edge_start = self._edge_ptr[idx].item()
        edge_end = self._edge_ptr[idx + 1].item()
        edge_ids = torch.arange(
            edge_start, edge_end, device=self.device, dtype=torch.long
        )

        # Get edges and remap to local indices
        local_edges = self.dist_edge_index[edge_ids]
        nodes_in_subgraph = local_edges.unique()

        # Vectorized node remapping - much faster than dictionary
        node_to_local = torch.zeros(
            nodes_in_subgraph.max().item() + 1, dtype=torch.long, device=self.device
        )
        node_to_local[nodes_in_subgraph] = torch.arange(
            nodes_in_subgraph.size(0), device=self.device
        )

        # Vectorized edge remapping - avoid list comprehensions
        src_local = node_to_local[local_edges[:, 0]]
        dst_local = node_to_local[local_edges[:, 1]]
        graph_edges = torch.stack([src_local, dst_local], dim=1)

        # Extract node features
        sub_x = self.feature_store["node", "x", None][nodes_in_subgraph]

        return {
            "x": sub_x,
            "edge_index": graph_edges,
            "y": y,
            "num_nodes": sub_x.size(0),
        }


def custom_collate_fn(batch):
    """Highly optimized custom collate function to batch graphs."""
    batch_size = len(batch)
    if batch_size == 0:
        return Data()

    # Extract data and compute dimensions in one pass
    x_list = [item["x"] for item in batch]
    edge_index_list = [item["edge_index"] for item in batch]
    y_list = [item["y"] for item in batch]
    num_nodes_list = [item["num_nodes"] for item in batch]

    # Pre-allocate tensors instead of using torch.cat for better performance
    total_nodes = sum(x.size(0) for x in x_list)
    total_edges = sum(edge.size(0) for edge in edge_index_list)
    feature_dim = x_list[0].size(1)
    device = x_list[0].device

    # Pre-allocate x_batch tensor
    x_batch = torch.empty(
        (total_nodes, feature_dim), dtype=x_list[0].dtype, device=device
    )

    # Pre-allocate y_batch tensor - handle zero-dimensional tensors efficiently
    y_batch = torch.empty(batch_size, dtype=y_list[0].dtype, device=device)

    # Pre-allocate batch_tensor
    batch_tensor = torch.empty(total_nodes, dtype=torch.long, device=device)

    # Fill tensors with vectorized operations
    node_offset = 0
    for i, (x, y, num_nodes) in enumerate(zip(x_list, y_list, num_nodes_list)):
        # Fill x_batch
        x_batch[node_offset : node_offset + num_nodes] = x

        # Fill y_batch - handle zero-dimensional tensors
        y_batch[i] = y.squeeze() if y.dim() > 0 else y

        # Fill batch_tensor
        batch_tensor[node_offset : node_offset + num_nodes] = i
        node_offset += num_nodes

    # Vectorized edge reindexing - much faster than loop
    if total_edges > 0:
        edge_index_final = torch.empty(
            (2, total_edges), dtype=torch.long, device=device
        )

        # Pre-compute node offsets using cumsum
        num_nodes_tensor = torch.tensor(num_nodes_list, dtype=torch.long, device=device)
        node_offsets = torch.cumsum(
            torch.cat(
                [torch.zeros(1, dtype=torch.long, device=device), num_nodes_tensor[:-1]]
            ),
            dim=0,
        )

        # Vectorized edge filling
        edge_offset = 0
        for i, edge_index in enumerate(edge_index_list):
            if edge_index.size(0) > 0:
                num_edges = edge_index.size(0)
                # Transpose and add offset in one operation
                edge_index_final[:, edge_offset : edge_offset + num_edges] = (
                    edge_index.t() + node_offsets[i]
                )
                edge_offset += num_edges
    else:
        edge_index_final = torch.empty((2, 0), dtype=torch.long, device=device)

    # Create PyG Data object
    batch_data = Data(
        x=x_batch, edge_index=edge_index_final, y=y_batch, batch=batch_tensor
    )
    batch_data.num_graphs = batch_size
    return batch_data


def setup_distributed():
    """Initialize distributed processing for cugraph-pyg."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GIN training with cugraph-pyg distributed backend"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ENZYMES",
        choices=[
            "MUTAG",
            "ENZYMES",
            "PROTEINS",
            "COLLAB",
            "IMDB-BINARY",
            "REDDIT-BINARY",
        ],
        help="Dataset name from TUDataset",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--hidden_channels", type=int, default=64, help="Number of hidden channels"
    )
    parser.add_argument(
        "--num_layers", type=int, default=5, help="Number of GIN layers"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument(
        "--train_split", type=float, default=0.9, help="Training split ratio"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for training"
    )
    parser.add_argument(
        "--data_root", type=str, default=None, help="Data root directory"
    )
    return parser.parse_args()


def get_dataset_stats(dataset_name):
    """Get dataset statistics for optimal hyperparameters"""
    stats = {
        "MUTAG": {"batch_size": 128, "hidden_channels": 32, "lr": 0.01},
        "ENZYMES": {"batch_size": 32, "hidden_channels": 64, "lr": 0.01},
        "PROTEINS": {"batch_size": 32, "hidden_channels": 64, "lr": 0.01},
        "COLLAB": {"batch_size": 32, "hidden_channels": 64, "lr": 0.01},
        "IMDB-BINARY": {"batch_size": 128, "hidden_channels": 64, "lr": 0.01},
        "REDDIT-BINARY": {"batch_size": 32, "hidden_channels": 64, "lr": 0.01},
    }
    return stats.get(
        dataset_name, {"batch_size": 32, "hidden_channels": 64, "lr": 0.01}
    )


def load_dataset_with_features(dataset_name, data_root=None):
    """Load dataset with automatic feature handling."""
    import os.path as osp

    if data_root is None:
        # Use a directory in the user's home where we have write permissions
        data_root = osp.join(osp.expanduser("~"), "data", "TU")

    # Check if dataset needs synthetic features
    temp_dataset = TUDataset(data_root, name=dataset_name)
    needs_features = temp_dataset.num_features == 0

    if needs_features:
        # Use OneHotDegree transform for datasets without node features
        # Handle REDDIT-BINARY which has extremely high node degrees
        if dataset_name == "REDDIT-BINARY":
            max_degree = 5000  # Much higher for REDDIT-BINARY
        else:
            max_degree = 1000
        transform = OneHotDegree(max_degree=max_degree, cat=False)
        dataset = TUDataset(data_root, name=dataset_name, transform=transform).shuffle()
    else:
        dataset = TUDataset(data_root, name=dataset_name).shuffle()

    return dataset


class GIN(torch.nn.Module):
    """Graph Isomorphism Network for graph classification."""

    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.5
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP(
            [hidden_channels, hidden_channels, out_channels], norm=None, dropout=dropout
        )

    def forward(self, x, edge_index, batch, batch_size):
        """Forward pass with graph-level pooling."""
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch, size=batch_size)
        return self.mlp(x)


def load_data(dataset_name, data_root, device, device_id):
    """Load dataset and set up distributed storage."""
    dataset = load_dataset_with_features(dataset_name, data_root)
    data = Batch.from_data_list(dataset)

    # Use PyG's built-in ptr tensor for edge pointers - much smarter!
    edge_index = data.edge_index.t()  # shape [E, 2]

    # PyG automatically computes edge_ptr as batch.ptr
    edge_ptr = data.ptr

    # Create distributed storage
    dist_edge_index = DistTensor.from_tensor(tensor=edge_index)
    feature_store = FeatureStore()
    feature_store["node", "x", None] = data.x
    feature_store["graph", "y", None] = data.y

    num_features = data.x.size(1)
    num_classes = int(data.y.max().item()) + 1

    return (
        (feature_store, dist_edge_index),
        edge_ptr,
        len(dataset),
        num_features,
        num_classes,
    )


def train(model, loader, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch, batch.num_graphs)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(model, loader, device):
    """Evaluate the model on the given loader."""
    model.eval()
    total_correct = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch, batch.num_graphs)
        pred = out.argmax(dim=-1)
        total_correct += int((pred == batch.y).sum())
    return total_correct / len(loader.dataset)


def main():
    """Main training function."""
    setup_distributed()
    args = parse_args()

    # Setup device
    device = (
        torch.device("cuda:0")
        if args.device == "cuda" and torch.cuda.is_available()
        else torch.device(args.device)
    )
    device_id = 0

    # Get optimal hyperparameters for the dataset
    dataset_stats = get_dataset_stats(args.dataset)

    # Override with command line arguments if provided
    batch_size = (
        args.batch_size if args.batch_size != 32 else dataset_stats["batch_size"]
    )
    hidden_channels = (
        args.hidden_channels
        if args.hidden_channels != 64
        else dataset_stats["hidden_channels"]
    )
    lr = args.lr if args.lr != 0.01 else dataset_stats["lr"]

    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {batch_size}")
    print(f"Hidden channels: {hidden_channels}")
    print(f"Learning rate: {lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {device}, CUDA available: {torch.cuda.is_available()}")

    # Load data and create distributed storage
    (
        (feature_store, dist_edge_index),
        edge_ptr,
        num_graphs,
        num_features,
        num_classes,
    ) = load_data(args.dataset, args.data_root, device, device_id)

    print(f"Number of features: {num_features}")
    print(f"Number of classes: {num_classes}")

    # Create train/test split
    train_size = int(args.train_split * num_graphs)

    train_graph_indices = list(range(0, train_size))
    test_graph_indices = list(range(train_size, num_graphs))

    print(f"Dataset size: {num_graphs}")
    print(f"Training samples: {len(train_graph_indices)}")
    print(f"Test samples: {len(test_graph_indices)}")

    # Create datasets and loaders
    train_dataset = DistTensorGraphDataset(
        dist_edge_index, feature_store, device, edge_ptr, train_graph_indices
    )
    test_dataset = DistTensorGraphDataset(
        dist_edge_index, feature_store, device, edge_ptr, test_graph_indices
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
    )

    # Create model and optimizer
    model = GIN(
        num_features, hidden_channels, num_classes, args.num_layers, args.dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    train_times = []
    test_times = []
    total_times = []
    for epoch in range(1, args.epochs + 1):
        # Training time
        train_start = time.time()
        loss = train(model, train_loader, optimizer, device)
        train_time = time.time() - train_start

        # Testing time
        test_start = time.time()
        train_acc = test(model, train_loader, device)
        test_acc = test(model, test_loader, device)
        test_time = time.time() - test_start

        total_time = train_time + test_time
        train_times.append(train_time)
        test_times.append(test_time)
        total_times.append(total_time)

        print(
            f"Epoch: {epoch:03d}, Loss: {loss:.4f},"
            f" Train: {train_acc:.4f}, Test: {test_acc:.4f}"
        )
        print(
            f"  Train Time: {train_time:.4f}s, "
            f"Test Time: {test_time:.4f}s, Total: {total_time:.4f}s"
        )

    print(
        f"Training - Median: {torch.tensor(train_times).median():.4f}s,"
        f" Average: {torch.tensor(train_times).mean():.4f}s"
    )
    print(
        f"Testing  - Median: {torch.tensor(test_times).median():.4f}s,"
        f" Average: {torch.tensor(test_times).mean():.4f}s"
    )
    print(
        f"Total    - Median: {torch.tensor(total_times).median():.4f}s,"
        f" Average: {torch.tensor(total_times).mean():.4f}s"
    )
    print(f"Final Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
    if dist.is_initialized():
        dist.destroy_process_group()
