import os
import warnings
from argparse import ArgumentParser
from datetime import timedelta

import json

import torch
import torch.nn.functional as F

from torch.nn import Linear


from tqdm import tqdm

from torch_geometric import EdgeIndex
from torch_geometric.datasets import MovieLens

from torch_geometric.nn import SAGEConv
from torch_geometric.data import HeteroData

from pylibwholegraph.torch.initialize import (
    init as wm_init,
    finalize as wm_finalize,
)

from sklearn.metrics import roc_auc_score


def init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id):
    import rmm

    rmm.reinitialize(
        devices=local_rank,
        managed_memory=False,
        pool_allocator=False,
    )

    import cupy

    cupy.cuda.Device(local_rank).use()
    from rmm.allocators.cupy import rmm_cupy_allocator

    cupy.cuda.set_allocator(rmm_cupy_allocator)

    from cugraph.testing.mg_utils import enable_spilling

    enable_spilling()

    torch.cuda.set_device(local_rank)

    from cugraph.gnn import cugraph_comms_init

    cugraph_comms_init(
        rank=global_rank, world_size=world_size, uid=cugraph_id, device=local_rank
    )

    wm_init(global_rank, world_size, local_rank, torch.cuda.device_count())


def write_edges(edge_index, path):
    world_size = torch.distributed.get_world_size()

    os.makedirs(path, exist_ok=True)
    for (r, e) in enumerate(torch.tensor_split(edge_index, world_size, dim=1)):
        rank_path = os.path.join(path, f"rank={r}.pt")
        torch.save(
            e.clone(),
            rank_path,
        )


def cugraph_pyg_from_heterodata(data, wg_mem_type):
    from cugraph_pyg.data import GraphStore, WholeFeatureStore

    graph_store = GraphStore(is_multi_gpu=True)
    feature_store = WholeFeatureStore(memory_type=wg_mem_type)

    graph_store[
        ("user", "rates", "movie"),
        "coo",
        False,
        (data["user"].num_nodes, data["movie"].num_nodes),
    ] = data["user", "rates", "movie"].edge_index

    graph_store[
        ("movie", "rev_rates", "user"),
        "coo",
        False,
        (data["movie"].num_nodes, data["user"].num_nodes),
    ] = data["movie", "rev_rates", "user"].edge_index

    feature_store["user", "x", None] = data["user"].x
    feature_store["movie", "x", None] = data["movie"].x

    return feature_store, graph_store


def preprocess_and_partition(data, edge_path, features_path, meta_path):
    world_size = torch.distributed.get_world_size()

    # Only use edges with high ratings (>= 4):
    mask = data["user", "rates", "movie"].edge_label >= 4
    data["user", "movie"].edge_index = data["user", "movie"].edge_index[:, mask]
    data["user", "movie"].time = data["user", "movie"].time[mask]
    del data["user", "movie"].edge_label  # Drop rating information from graph.

    # Perform a temporal link-level split into training and test edges:
    time = data["user", "movie"].time
    perm = time.argsort()

    # Reorder the edge index so the time split is even
    data["user", "movie"] = data["user", "movie"].edge_index[:, perm]

    # Reserve first 80% for train, last 20% for test
    off = int(0.8 * perm.numel())
    ei = {
        "train": data["user", "movie"].edge_index[:, :off],
        "test": data["user", "movie"].edge_index[:, off:],
    }

    print("Writing edges...")
    user_movie_edge_path = os.path.join(edge_path, "user_movie")
    for d, eid in ei.items():
        d_path = os.path.join(user_movie_edge_path, d)
        write_edges(eid, d_path)

    print("Writing features...")
    movie_path = os.path.join(features_path, "movie")
    os.makedirs(
        movie_path,
        exist_ok=True,
    )
    for r, fx in enumerate(torch.tensor_split(data["movie"].x, world_size)):
        torch.save(
            fx,
            os.path.join(movie_path, f"rank={r}.pt"),
        )

    print("Writing metadata...")
    meta = {
        "num_nodes": {
            "movie": data["movie"].num_nodes,
            "user": data["user"].num_nodes,
        }
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)


def load_partitions(edge_path, features_path, meta_path):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    data = HeteroData()

    # Load metadata
    print("Loading metadata...")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    data["user"].num_nodes = meta["num_nodes"]["user"]
    data["movie"].num_nodes = meta["num_nodes"]["movie"]
    data["user"].x = (
        torch.tensor_split(
            torch.eye(data["user"].num_nodes, dtype=torch.float32), world_size
        )[rank]
        .detach()
        .clone()
    )
    data["movie"].x = torch.load(
        os.path.join(features_path, "movie", f"rank={rank}.pt"),
        weights_only=True,
    )

    # T.ToUndirected() will not work here because we are working with
    # partitioned data.  The number of nodes will not match.

    print("Loading user->movie edge index...")
    ei = {}
    for d in {"train", "test"}:
        ei[d] = torch.load(
            os.path.join(edge_path, "user_movie", d, f"rank={rank}.pt"),
            weights_only=True,
        )

    data["user", "rates", "movie"].edge_index = torch.concat(
        [
            ei["train"],
            ei["test"],
        ],
        dim=1,
    )

    label_dict = {
        "train": torch.randperm(ei["train"].shape[1]),
        "test": torch.randperm(ei["test"].shape[1]) + ei["train"].shape[1],
    }

    data["movie", "rev_rates", "user"].edge_index = torch.stack(
        [
            data["user", "rates", "movie"].edge_index[1],
            data["user", "rates", "movie"].edge_index[0],
        ]
    )

    print(f"Finished loading graph data on rank {rank}")
    return data, label_dict


class Encoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels)
        self.lin1 = Linear(hidden_channels, out_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        user_x = self.conv1(
            (x_dict["movie"], x_dict["user"]),
            edge_index_dict["movie", "rev_rates", "user"],
        ).relu()

        movie_x = self.conv2(
            (x_dict["user"], x_dict["movie"]), edge_index_dict["user", "rates", "movie"]
        ).relu()

        user_x = self.conv3(
            (movie_x, user_x), edge_index_dict["movie", "rev_rates", "user"]
        ).relu()

        return {
            "user": self.lin1(user_x),
            "movie": self.lin2(movie_x),
        }


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, x_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat(
            [
                x_dict["user"][row],
                x_dict["movie"][col],
            ],
            dim=-1,
        )

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, metadata):
        super().__init__()
        self.encoder = Encoder(hidden_channels, hidden_channels)
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, num_samples):
        x_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(
            x_dict, edge_index_dict["user", "rates", "movie"][:, :num_samples]
        )


def train(train_loader, model, optimizer):
    model.train()

    total_loss = total_examples = 0
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch["user", "rates", "movie"].edge_label.shape[0],
        )

        y = batch["user", "rates", "movie"].edge_label

        loss = F.binary_cross_entropy_with_logits(out, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * y.numel()
        total_examples += y.numel()

    return total_loss / total_examples


@torch.no_grad()
def test(test_loader, model):
    model.eval()

    preds = []
    targets = []
    for batch in test_loader:
        batch = batch.to(device)
        pred = (
            model(
                batch.x_dict,
                batch.edge_index_dict,
                batch["user", "rates", "movie"].edge_label.shape[0],
            )
            .sigmoid()
            .view(-1)
            .cpu()
        )

        target = batch["user", "rates", "movie"].edge_label.long().cpu()

        preds.append(pred)
        targets.append(target)

    pred = torch.cat(preds, dim=0).numpy()
    target = torch.cat(targets, dim=0).numpy()

    return roc_auc_score(target, pred)


if __name__ == "__main__":
    if "LOCAL_RANK" not in os.environ:
        warnings.warn("This script should be run with 'torchrun`.  Exiting.")
        exit()

    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--dataset_root", type=str, default="datasets")
    parser.add_argument("--skip_partition", action="store_true")
    parser.add_argument("--wg_mem_type", type=str, default="distributed")
    args = parser.parse_args()

    dataset_name = "movielens"

    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=3600))
    world_size = torch.distributed.get_world_size()
    global_rank = torch.distributed.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(local_rank)

    if global_rank == 0:
        from rmm.allocators.torch import rmm_torch_allocator

        torch.cuda.change_current_allocator(rmm_torch_allocator)

    # Create the uid needed for cuGraph comms
    if global_rank == 0:
        from cugraph.gnn import (
            cugraph_comms_create_unique_id,
        )

        cugraph_id = [cugraph_comms_create_unique_id()]
    else:
        cugraph_id = [None]
    torch.distributed.broadcast_object_list(cugraph_id, src=0, device=device)
    cugraph_id = cugraph_id[0]

    init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id)

    # Split the data
    edge_path = os.path.join(args.dataset_root, dataset_name + "_eix_part")
    features_path = os.path.join(args.dataset_root, dataset_name + "_feat")
    meta_path = os.path.join(args.dataset_root, dataset_name + "_meta.json")

    if not args.skip_partition and global_rank == 0:
        print("Partitioning data...")

        dataset = MovieLens(args.dataset_root, model_name="all-MiniLM-L6-v2")
        data = dataset[0]

        preprocess_and_partition(
            data, edge_path=edge_path, features_path=features_path, meta_path=meta_path
        )
        print("Data partitioning complete!")

    torch.distributed.barrier()
    data, label_dict = load_partitions(
        edge_path=edge_path, features_path=features_path, meta_path=meta_path
    )
    torch.distributed.barrier()

    feature_store, graph_store = cugraph_pyg_from_heterodata(
        data, wg_mem_type=args.wg_mem_type
    )
    eli_train = data["user", "rates", "movie"].edge_index[:, label_dict["train"]]
    eli_test = data["user", "rates", "movie"].edge_index[:, label_dict["test"]]
    num_nodes = {"user": data["user"].num_nodes, "movie": data["movie"].num_nodes}
    metadata = data.metadata()
    del data

    # TODO enable temporal sampling when it is available in cuGraph-PyG
    kwargs = dict(
        data=(feature_store, graph_store),
        num_neighbors={
            ("user", "rates", "movie"): [5, 5, 5],
            ("movie", "rev_rates", "user"): [5, 5, 5],
        },
        batch_size=256,
        # time_attr='time',
        shuffle=True,
        drop_last=True,
        # temporal_strategy='last',
    )

    from cugraph_pyg.loader import LinkNeighborLoader

    train_loader = LinkNeighborLoader(
        edge_label_index=(("user", "rates", "movie"), eli_train),
        # edge_label_time=time[train_index] - 1,  # No leakage.
        neg_sampling=dict(mode="binary", amount=2),
        **kwargs,
    )

    test_loader = LinkNeighborLoader(
        edge_label_index=(("user", "rates", "movie"), eli_test),
        neg_sampling=dict(mode="binary", amount=1),
        **kwargs,
    )

    sparse_size = (num_nodes["user"], num_nodes["movie"])
    test_edge_label_index = EdgeIndex(
        eli_test.to(device),
        sparse_size=sparse_size,
    ).sort_by("row")[0]
    test_exclude_links = EdgeIndex(
        eli_test.to(device),
        sparse_size=sparse_size,
    ).sort_by("row")[0]

    model = Model(hidden_channels=64, metadata=metadata).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_loss = train(train_loader, model, optimizer)
        print(f"Epoch: {epoch:02d}, Loss: {train_loss:.4f}")
        auc = test(test_loader, model)
        print(f"Test AUC: {auc:.4f} ")

    from cugraph.gnn import cugraph_comms_shutdown

    cugraph_comms_shutdown()
    wm_finalize()
