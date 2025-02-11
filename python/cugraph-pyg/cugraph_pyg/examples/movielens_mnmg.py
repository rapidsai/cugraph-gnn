import os
import warnings
from argparse import ArgumentParser
from datetime import timedelta

import json

import torch
import torch.nn.functional as F
from tqdm import tqdm

from torch_geometric import EdgeIndex
from torch_geometric.datasets import MovieLens
from torch_geometric.metrics import (
    LinkPredMAP,
    LinkPredPrecision,
    LinkPredRecall,
)
from torch_geometric.nn import MIPSKNNIndex, SAGEConv, to_hetero
from torch_geometric.data import HeteroData

from pylibwholegraph.torch.initialize import (
    init as wm_init,
    finalize as wm_finalize,
)


def init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id):
    import rmm

    rmm.reinitialize(
        devices=local_rank,
        managed_memory=True,
        pool_allocator=True,
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

    return feature_store, graph_store


def preprocess_and_partition(data, edge_path, meta_path):
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

    print("Writing metadata...")
    meta = {
        "num_nodes": {
            "movie": data["movie"].num_nodes,
            "user": data["user"].num_nodes,
        }
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f)


def load_partitions(edge_path, meta_path):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    data = HeteroData()

    # Load metadata
    print("Loading metadata...")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    data["user"].num_nodes = meta["num_nodes"]["user"]
    data["movie"].num_nodes = meta["num_nodes"]["movie"]
    data["user"].x = torch.tensor_split(torch.eye(data["user"].num_nodes), world_size)[
        rank
    ]

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


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        return x


class InnerProductDecoder(torch.nn.Module):
    def forward(self, x_dict, edge_label_index):
        x_src = x_dict["user"][edge_label_index[0]]
        x_dst = x_dict["movie"][edge_label_index[1]]
        return (x_src * x_dst).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNN(hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr="sum")
        self.decoder = InnerProductDecoder()

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        x_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(x_dict, edge_label_index)


def train():
    model.train()

    total_loss = total_examples = 0
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(
            batch.x_dict,
            batch.edge_index_dict,
            batch["user", "movie"].edge_label_index,
        )
        y = batch["user", "movie"].edge_label

        loss = F.binary_cross_entropy_with_logits(out, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * y.numel()
        total_examples += y.numel()

    return total_loss / total_examples


@torch.no_grad()
def test(edge_label_index, exclude_links):
    model.eval()

    dst_embs = []
    for batch in dst_loader:  # Collect destination node/movie embeddings:
        batch = batch.to(device)
        emb = model.encoder(batch.x_dict, batch.edge_index_dict)["movie"]
        emb = emb[: batch["movie"].batch_size]
        dst_embs.append(emb)
    dst_emb = torch.cat(dst_embs, dim=0)
    del dst_embs

    # Instantiate k-NN index based on maximum inner product search (MIPS):
    mips = MIPSKNNIndex(dst_emb)

    # Initialize metrics:
    map_metric = LinkPredMAP(k=args.npred).to(device)
    precision_metric = LinkPredPrecision(k=args.npred).to(device)
    recall_metric = LinkPredRecall(k=args.npred).to(device)

    num_processed = 0
    for batch in src_loader:  # Collect source node/user embeddings:
        batch = batch.to(device)

        # Compute user embeddings:
        emb = model.encoder(batch.x_dict, batch.edge_index_dict)["user"]
        emb = emb[: batch["user"].batch_size]

        # Filter labels/exclusion by current batch:
        _edge_label_index = edge_label_index.sparse_narrow(
            dim=0,
            start=num_processed,
            length=emb.size(0),
        )
        _exclude_links = exclude_links.sparse_narrow(
            dim=0,
            start=num_processed,
            length=emb.size(0),
        )
        num_processed += emb.size(0)

        # Perform MIPS search:
        _, pred_index_mat = mips.search(emb, args.npred, _exclude_links)

        # Update retrieval metrics:
        map_metric.update(pred_index_mat, _edge_label_index)
        precision_metric.update(pred_index_mat, _edge_label_index)
        recall_metric.update(pred_index_mat, _edge_label_index)

    return (
        float(map_metric.compute()),
        float(precision_metric.compute()),
        float(recall_metric.compute()),
    )


if __name__ == "__main__":
    if "LOCAL_RANK" not in os.environ:
        warnings.warn("This script should be run with 'torchrun`.  Exiting.")
        exit()

    parser = ArgumentParser()
    parser.add_argument("--npred", type=int, default=20, help="Number of predictions")
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
    meta_path = os.path.join(args.dataset_root, dataset_name + "_meta.json")

    if not args.skip_partition and global_rank == 0:
        print("Partitioning data...")

        dataset = MovieLens(args.dataset_root, model_name="all-MiniLM-L6-v2")
        data = dataset[0]

        preprocess_and_partition(data, edge_path=edge_path, meta_path=meta_path)
        print("Data partitioning complete!")

    torch.distributed.barrier()
    data, label_dict = load_partitions(edge_path=edge_path, meta_path=meta_path)
    torch.distributed.barrier()

    feature_store, graph_store = cugraph_pyg_from_heterodata(
        data, wg_mem_type=args.wg_mem_type
    )
    eli_train = data["user", "rates", "movie"].edge_index[:, label_dict["train"]]
    eli_test = data["user", "rates", "movie"].edge_index[:, label_dict["test"]]
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

    from cugraph_pyg.loader import NeighborLoader, LinkNeighborLoader

    train_loader = LinkNeighborLoader(
        edge_label_index=(("user", "rates", "movie"), eli_train),
        # edge_label_time=time[train_index] - 1,  # No leakage.
        neg_sampling=dict(mode="binary", amount=2),
        **kwargs,
    )

    src_loader = NeighborLoader(
        input_nodes="user",
        # input_time=(time[test_index].min() - 1).repeat(data['user'].num_nodes),
        **kwargs,
    )
    dst_loader = NeighborLoader(
        input_nodes="movie",
        # input_time=(time[test_index].min() - 1).repeat(data['movie'].num_nodes),
        **kwargs,
    )

    sparse_size = (data["user"].num_nodes, data["movie"].num_nodes)
    test_edge_label_index = EdgeIndex(
        eli_test.to(device),
        sparse_size=sparse_size,
    ).sort_by("row")[0]
    test_exclude_links = EdgeIndex(
        eli_test.to(device),
        sparse_size=sparse_size,
    ).sort_by("row")[0]

    model = Model(hidden_channels=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.num_epochs):
        train_loss = train()
        print(f"Epoch: {epoch:02d}, Loss: {train_loss:.4f}")
        val_map, val_precision, val_recall = test(
            test_edge_label_index,
            test_exclude_links,
        )
        print(
            f"Test MAP@{args.npred}: {val_map:.4f}, "
            f"Test Precision@{args.npred}: {val_precision:.4f}, "
            f"Test Recall@{args.npred}: {val_recall:.4f}"
        )

    from cugraph.gnn import cugraph_comms_shutdown

    cugraph_comms_shutdown()
    wm_finalize()
    torch.distributed.destroy_process_group()
