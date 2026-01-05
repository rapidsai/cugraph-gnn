# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import warnings

import torch
from torch.nn import Linear, Dropout, LayerNorm
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

from torch_geometric.nn import TransformerConv, SAGEConv, to_hetero, Sequential

import pylibwholegraph.torch as wgth

from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])


class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, edge_attr_dim, heads=1):
        super().__init__()

        self.conv1 = TransformerConv(
            in_channels,
            hidden_channels,
            edge_dim=edge_attr_dim,
            concat=False,
            heads=heads,
        )
        self.conv2 = TransformerConv(
            hidden_channels,
            hidden_channels,
            edge_dim=edge_attr_dim,
            concat=False,
            heads=heads,
        )

        self.norm1 = LayerNorm(hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.norm2 = LayerNorm(hidden_channels)

        self.dropout = Dropout(p=0.5)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr) + self.lin1(x)
        x = self.norm1(x)
        x = self.dropout(x)
        x = x.relu()

        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = self.dropout(x)
        x = x.relu()

        x = self.lin2(x).relu()

        return F.normalize(x, p=2, dim=-1)


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        return (x1 * x2).sum(dim=-1)


class Classifier(torch.nn.Module):
    def __init__(
        self,
        hidden_channels,
        num_features,
        num_nodes,
        edge_attr_dim,
        metadata,
        learn_embeddings=False,
    ):
        super().__init__()

        self.learn_embeddings = learn_embeddings
        self.hidden_channels = hidden_channels

        self.paper_lin = Linear(num_features["paper"], hidden_channels)
        self.paper_norm = LayerNorm(hidden_channels)

        self.embeddings = {}
        if self.learn_embeddings:
            global_comm = wgth.get_global_communicator()
            for node_type in sorted(num_nodes.keys()):
                wg_node_emb = wgth.create_embedding(
                    global_comm,
                    "distributed",
                    "cpu",
                    torch.float32,
                    [num_nodes[node_type], hidden_channels],
                    cache_policy=None,
                    random_init=True,
                )
                self.embeddings[node_type] = wgth.embedding.WholeMemoryEmbeddingModule(
                    wg_node_emb
                )
        else:
            self.mp = Sequential(
                "x, edge_index",
                [
                    (
                        SAGEConv((hidden_channels, hidden_channels), hidden_channels),
                        "x, edge_index -> x",
                    ),
                    LayerNorm(hidden_channels),
                    Dropout(p=0.5),
                    torch.nn.ReLU(inplace=True),
                ],
            )
            self.mp = to_hetero(self.mp, metadata=metadata, aggr="sum")

        self.encoder = Encoder(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            edge_attr_dim=edge_attr_dim,
        )
        self.encoder = to_hetero(self.encoder, metadata=metadata, aggr="sum")

        self.decoder = Decoder()

    def forward(self, batch, edge_attr):
        x_paper = self.paper_lin(batch["paper"].x)
        x_paper = self.paper_norm(x_paper)

        if self.learn_embeddings:
            x_dict = {
                "paper": x_paper + self.embeddings["paper"](batch["paper"].n_id),
                "author": self.embeddings["author"](batch["author"].n_id),
                "institution": self.embeddings["institution"](
                    batch["institution"].n_id
                ),
                "field_of_study": self.embeddings["field_of_study"](
                    batch["field_of_study"].n_id
                ),
            }
        else:
            # have to obtain embeddings through message passing
            x_dict = {
                "paper": x_paper,
                "author": torch.zeros(
                    batch["author"].n_id.numel(), self.hidden_channels, device="cuda"
                ),
                "institution": torch.zeros(
                    batch["institution"].n_id.numel(),
                    self.hidden_channels,
                    device="cuda",
                ),
                "field_of_study": torch.zeros(
                    batch["field_of_study"].n_id.numel(),
                    self.hidden_channels,
                    device="cuda",
                ),
            }
            x_dict = self.mp(x_dict, batch.edge_index_dict)

        x_dict = self.encoder(x_dict, batch.edge_index_dict, edge_attr)
        x_dict["paper"] += x_paper
        eli = batch["paper", "cites", "paper"].edge_label_index
        return self.decoder(x_dict["paper"][eli[0]], x_dict["paper"][eli[1]])


def init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id):
    import rmm

    rmm.reinitialize(
        devices=local_rank,
        managed_memory=True,
        pool_allocator=False,
    )

    from pylibwholegraph.torch.initialize import init as wm_init

    wm_init(global_rank, world_size, local_rank, torch.cuda.device_count())

    import cupy

    cupy.cuda.Device(local_rank).use()
    from rmm.allocators.cupy import rmm_cupy_allocator

    cupy.cuda.set_allocator(rmm_cupy_allocator)

    from pylibcugraph.comms import cugraph_comms_init

    cugraph_comms_init(
        rank=global_rank, world_size=world_size, uid=cugraph_id, device=local_rank
    )

    torch.cuda.set_device(local_rank)

@torch.no_grad()
def test(feature_store, test_loader, model, neg_ratio, eval_iter=100):
    model.eval()
    pred_true_pos = pred_false_pos = pred_true_neg = pred_false_neg = 0.0
    for i, batch in enumerate(test_loader):
        batch = batch.cuda()
        if i >= eval_iter:
            break

        y_pred = model(
            batch,
            {
                etype: feature_store[etype, "x", None][eid]
                for etype, eid in batch.e_id_dict.items()
            },
        )

        y_true = batch["paper", "cites", "paper"].edge_label.cuda()

        pred_true_pos += (
            ((y_pred > 0.5).float() == 1.0) & (y_true.float() == 1.0)
        ).sum()
        pred_false_pos += (
            ((y_pred > 0.5).float() == 1.0) & (y_true.float() == 0.0)
        ).sum()
        pred_true_neg += (
            ((y_pred <= 0.5).float() == 1.0) & (y_true.float() == 0.0)
        ).sum()
        pred_false_neg += (
            ((y_pred <= 0.5).float() == 1.0) & (y_true.float() == 1.0)
        ).sum()

    return pred_true_pos, pred_false_pos, pred_true_neg, pred_false_neg


@torch.enable_grad()
def train(
    feature_store,
    train_loader,
    model,
    optimizer,
    wm_optimizer,
    neg_ratio,
    lr=0.001,
    train_iter=100,
):
    model.train()
    total_loss = total_examples = 0
    global_rank = torch.distributed.get_rank()

    for i, batch in enumerate(train_loader):
        batch = batch.cuda()
        if i >= train_iter:
            break

        optimizer.zero_grad()
        out = model(
            batch,
            {
                etype: feature_store[etype, "x", None][eid]
                for etype, eid in batch.e_id_dict.items()
            },
        )

        loss = F.binary_cross_entropy_with_logits(
            out, batch["paper", "cites", "paper"].edge_label.cuda()
        )
        loss.backward()
        optimizer.step()
        if wm_optimizer:
            wm_optimizer.step(lr)
        total_loss += loss.item() * out.numel()
        total_examples += out.numel()

        if i % 10 == 0 and global_rank == 0:
            print(f"iter {i}, loss {loss.item():.4f}")

    return total_loss / total_examples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--train_iter", type=int, default=4096)
    parser.add_argument("--eval_iter", type=int, default=1024)
    parser.add_argument("--learn_embeddings", action="store_true")
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--betweenness_k", type=int, default=100)
    parser.add_argument("--betweenness_seed", type=int, default=62)
    parser.add_argument("--neg_ratio", type=int, default=1)
    parser.add_argument("--dataset_root", type=str, default="datasets")
    parser.add_argument("--output_dir", type=str, default="embeddings")
    args = parser.parse_args()

    torch.distributed.init_process_group(backend="nccl")

    if "LOCAL_RANK" not in os.environ:
        warnings.warn("This script should be run with 'torchrun`.  Exiting.")
        exit()

    global_rank = torch.distributed.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(local_rank)
    world_size = torch.distributed.get_world_size()

    if global_rank == 0:
        from pylibcugraph.comms import (
            cugraph_comms_create_unique_id,
        )

        cugraph_id = [cugraph_comms_create_unique_id()]
    else:
        cugraph_id = [None]
    torch.distributed.broadcast_object_list(cugraph_id, src=0, device=device)
    cugraph_id = cugraph_id[0]
    init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id)

    from cugraph_pyg.data import FeatureStore, GraphStore

    feature_store = FeatureStore()
    graph_store = GraphStore()

    torch.distributed.barrier()
    if global_rank == 0:
        print("loading dataset...")
        from ogb.nodeproppred import PygNodePropPredDataset

        dataset = PygNodePropPredDataset(name="ogbn-mag", root=args.dataset_root)
        data = dataset[0]

        # have to use "dict" here because OGB doesn't use the updated PyG API
        ei = data.edge_index_dict
        num_nodes = data.num_nodes_dict

        # add nodes
        print("adding nodes...")
        node_counts = torch.tensor(
            [
                num_nodes["paper"],
                num_nodes["author"],
                num_nodes["institution"],
                num_nodes["field_of_study"],
            ],
            device="cuda",
            dtype=torch.int64,
        )
        torch.distributed.broadcast(node_counts, src=0)

        # add edges
        print("adding edges...")
        graph_store[
            ("paper", "cites", "paper"),
            "coo",
            False,
            (num_nodes["paper"], num_nodes["paper"]),
        ] = ei["paper", "cites", "paper"]
        graph_store[
            ("author", "writes", "paper"),
            "coo",
            False,
            (num_nodes["author"], num_nodes["paper"]),
        ] = ei["author", "writes", "paper"]
        graph_store[
            ("author", "affiliated_with", "institution"),
            "coo",
            False,
            (num_nodes["author"], num_nodes["institution"]),
        ] = ei["author", "affiliated_with", "institution"]
        graph_store[
            ("paper", "has_topic", "field_of_study"),
            "coo",
            False,
            (num_nodes["paper"], num_nodes["field_of_study"]),
        ] = ei["paper", "has_topic", "field_of_study"]

        # add reverse edges
        print("adding reverse edges...")
        for edge_type in [
            ("paper", "cites", "paper"),
            ("author", "writes", "paper"),
            ("author", "affiliated_with", "institution"),
            ("paper", "has_topic", "field_of_study"),
        ]:
            graph_store[
                (edge_type[2], "rev_" + edge_type[1], edge_type[0]),
                "coo",
                False,
                (num_nodes[edge_type[2]], num_nodes[edge_type[0]]),
            ] = ei[edge_type].flip(0)

        # add features
        print("adding features...")
        feature_store["paper", "x", None] = data.x_dict["paper"]

        y = data.y_dict["paper"]
        del data
        del dataset
    else:
        from cugraph_pyg.tensor import empty

        # add nodes
        num_nodes = {}
        node_counts = torch.tensor([0, 0, 0, 0], device="cuda", dtype=torch.int64)
        torch.distributed.broadcast(node_counts, src=0)
        num_nodes["paper"] = node_counts[0]
        num_nodes["author"] = node_counts[1]
        num_nodes["institution"] = node_counts[2]
        num_nodes["field_of_study"] = node_counts[3]

        # add edges
        graph_store[
            ("paper", "cites", "paper"),
            "coo",
            False,
            (num_nodes["paper"], num_nodes["paper"]),
        ] = empty(dim=2)
        graph_store[
            ("author", "writes", "paper"),
            "coo",
            False,
            (num_nodes["author"], num_nodes["paper"]),
        ] = empty(dim=2)
        graph_store[
            ("author", "affiliated_with", "institution"),
            "coo",
            False,
            (num_nodes["author"], num_nodes["institution"]),
        ] = empty(dim=2)
        graph_store[
            ("paper", "has_topic", "field_of_study"),
            "coo",
            False,
            (num_nodes["paper"], num_nodes["field_of_study"]),
        ] = empty(dim=2)

        # add reverse edges
        for edge_type in [
            ("paper", "cites", "paper"),
            ("author", "writes", "paper"),
            ("author", "affiliated_with", "institution"),
            ("paper", "has_topic", "field_of_study"),
        ]:
            graph_store[
                (edge_type[2], "rev_" + edge_type[1], edge_type[0]),
                "coo",
                False,
                (num_nodes[edge_type[2]], num_nodes[edge_type[0]]),
            ] = empty(dim=2)

        # add features
        feature_store["paper", "x", None] = empty(dim=2)

    torch.distributed.barrier()

    from pylibcugraph import betweenness_centrality

    print("calculating betweenness centrality...")
    vx, vy = betweenness_centrality(
        resource_handle=graph_store._resource_handle,
        graph=graph_store._graph,
        k=args.betweenness_k,
        random_state=args.betweenness_seed + global_rank,
        normalized=True,
        include_endpoints=False,
        do_expensive_check=False,
    )

    vx = torch.as_tensor(vx, device="cuda")
    vy = torch.as_tensor(vy, device="cuda")

    offsets = torch.tensor(
        sorted(graph_store._vertex_offsets.values()),
        device="cpu",
        dtype=torch.int64,
    )
    vtypes = sorted(graph_store._vertex_offsets.keys())

    print(f"rank {global_rank}, offsets {offsets}")
    for i, vtype in enumerate(vtypes):
        if i == len(vtypes) - 1:
            f = vx >= offsets[i]
        else:
            f = (vx >= offsets[i]) & (vx < offsets[i + 1])

        bcx = vx[f] - offsets[i]
        bcy = vy[f].to(torch.float32)
        feature_store[vtype, "bc", bcx] = bcy

    print("updating feature store with betweeness centralities...")
    for etype in graph_store.get_all_edge_attrs():
        stype, _, dtype = etype.edge_type
        src, dst = graph_store[etype]
        # bug in torch_geometric EdgeIndex requires we reconstruct the tensors
        src = src.clone().detach()
        dst = dst.clone().detach()

        feature_store[etype.edge_type, "x", None] = (
            feature_store[stype, "bc", None][src]
            + feature_store[dtype, "bc", None][dst]
        ).reshape((-1, 1)) / 2.0

    print("training model...")

    model = Classifier(
        hidden_channels=args.hidden_channels,
        num_features={
            "paper": feature_store["paper", "x", None].shape[1],
            "author": 0,
            "institution": 0,
            "field_of_study": 0,
        },
        num_nodes=num_nodes,
        edge_attr_dim=1,
        metadata=(
            vtypes,
            [etype.edge_type for etype in graph_store.get_all_edge_attrs()],
        ),
        learn_embeddings=args.learn_embeddings,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.learn_embeddings:
        wm_optimizer = wgth.create_wholememory_optimizer(
            [
                model.embeddings[node_type].wm_embedding
                for node_type in sorted(num_nodes.keys())
            ],
            "adam",
            {},
        )
    else:
        wm_optimizer = None

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    assigned_edges = graph_store[("paper", "cites", "paper"), "coo", None]
    mask = (torch.rand(assigned_edges.shape[1]) < 0.8).to(torch.bool).to(device)
    train_edges = assigned_edges[:, mask]
    test_edges = assigned_edges[:, ~mask]

    train_sz = torch.tensor([train_edges.shape[1]], device="cuda", dtype=torch.int64)
    test_sz = torch.tensor([test_edges.shape[1]], device="cuda", dtype=torch.int64)
    torch.distributed.all_reduce(train_sz, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(test_sz, op=torch.distributed.ReduceOp.MIN)
    train_edges = train_edges[:, :train_sz]
    test_edges = test_edges[:, :test_sz]

    from cugraph_pyg.loader import LinkNeighborLoader

    train_loader = LinkNeighborLoader(
        data=(feature_store, graph_store),
        num_neighbors={
            etype.edge_type: [5] * 2 for etype in graph_store.get_all_edge_attrs()
        },
        edge_label_index=(("paper", "cites", "paper"), train_edges),
        neg_sampling=dict(mode="binary", amount=args.neg_ratio),
        batch_size=256,
        shuffle=True,
        drop_last=True,
    )

    test_loader = LinkNeighborLoader(
        data=(feature_store, graph_store),
        num_neighbors={
            etype.edge_type: [5] * 2 for etype in graph_store.get_all_edge_attrs()
        },
        edge_label_index=(("paper", "cites", "paper"), test_edges),
        neg_sampling=dict(mode="binary", amount=1),
        batch_size=256,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(1, args.epochs + 1):
        train(
            feature_store,
            train_loader,
            model,
            optimizer,
            wm_optimizer,
            args.neg_ratio,
            lr=args.lr,
            train_iter=args.train_iter,
        )
        pred_true_pos, pred_false_pos, pred_true_neg, pred_false_neg = test(
            feature_store, test_loader, model, args.neg_ratio, eval_iter=args.eval_iter
        )

        torch.distributed.all_reduce(pred_true_pos, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(pred_false_pos, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(pred_true_neg, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(pred_false_neg, op=torch.distributed.ReduceOp.SUM)

        total_examples = (
            pred_true_pos.item()
            + pred_false_pos.item()
            + pred_true_neg.item()
            + pred_false_neg.item()
        )
        pred_true_pos = int(pred_true_pos.item())
        pred_false_pos = int(pred_false_pos.item())
        pred_true_neg = int(pred_true_neg.item())
        pred_false_neg = int(pred_false_neg.item())

        if global_rank == 0:
            print(
                f"epoch {epoch}, acc (link pred): {(pred_true_pos + pred_true_neg) / total_examples:.4f}"
            )
            print(
                f"confusion (link pred):\nTP: {pred_true_pos}\tFN: {pred_false_neg}\nFP: {pred_false_pos}\tTN: {pred_true_neg}"
            )

    local_x0 = feature_store["paper", "x", None].get_local_tensor()

    ix_start = torch.tensor([local_x0.shape[0]], device="cuda", dtype=torch.int64)
    ixa = torch.empty((world_size,), device="cuda", dtype=torch.int64)
    torch.distributed.all_gather_into_tensor(ixa, ix_start)
    ixa = ixa.cumsum(0)
    ix_start = int(ixa[global_rank - 1]) if global_rank > 0 else 0
    ix_end = int(ix_start + local_x0.shape[0])

    if args.learn_embeddings:
        local_x1 = (
            model.module.embeddings["paper"]
            .wm_embedding.get_embedding_tensor()
            .get_local_tensor()[0]
        )
    else:
        from cugraph_pyg.loader import NeighborLoader

        local_papers = torch.arange(ix_start, ix_end, device="cuda", dtype=torch.int64)
        print(
            f"rank {global_rank}, local_papers {local_papers}, {local_papers.min()}, {local_papers.max()}"
        )
        ex_loader = NeighborLoader(
            data=(feature_store, graph_store),
            num_neighbors={
                etype.edge_type: [5] * 2 for etype in graph_store.get_all_edge_attrs()
            },
            input_nodes=("paper", local_papers),
            batch_size=256,
            shuffle=True,
            drop_last=False,
        )

        feature_store["paper", "x1", None] = torch.empty(
            (local_papers.shape[0], model.module.hidden_channels), device="cuda"
        )
        for batch in ex_loader:
            batch = batch.cuda()
            # have to obtain embeddings through message passing
            x_paper = model.module.paper_lin(batch["paper"].x)
            x_paper = model.module.paper_norm(x_paper)
            x_dict = {
                "paper": x_paper,
                "author": torch.zeros(
                    batch["author"].n_id.numel(),
                    model.module.hidden_channels,
                    device="cuda",
                ),
                "institution": torch.zeros(
                    batch["institution"].n_id.numel(),
                    model.module.hidden_channels,
                    device="cuda",
                ),
                "field_of_study": torch.zeros(
                    batch["field_of_study"].n_id.numel(),
                    model.module.hidden_channels,
                    device="cuda",
                ),
            }
            x_dict = model.module.mp(x_dict, batch.edge_index_dict)
            x_dict = model.module.encoder(
                x_dict,
                batch.edge_index_dict,
                edge_attr={
                    etype: feature_store[etype, "x", None][eid]
                    for etype, eid in batch.e_id_dict.items()
                },
            )
            feature_store["paper", "x1", None][
                batch["paper"].n_id[: batch["paper"].batch_size]
            ] = (
                x_dict["paper"][: batch["paper"].batch_size]
                + x_paper[: batch["paper"].batch_size]
            )
        local_x1 = feature_store["paper", "x1", None][local_papers]
    import cupy

    print("Finished computing embeddings, writing output embeddings to parquet...")
    local_x = cupy.asarray(torch.concat([local_x0, local_x1], dim=1))
    import cudf

    os.makedirs(os.path.join(args.output_dir, "x"), exist_ok=True)
    df = cudf.DataFrame(
        local_x,
        columns=[f"x_{i}" for i in range(local_x.shape[1])],
        index=cupy.arange(ix_start, ix_end, dtype="int64"),
    )
    df.to_parquet(os.path.join(args.output_dir, "x", f"x_{global_rank}.parquet"))
    from pylibcugraph.comms import cugraph_comms_shutdown

    cugraph_comms_shutdown()

    torch.distributed.barrier()
    from pylibwholegraph.torch.initialize import finalize as wm_finalize

    wm_finalize()  # will also destroy the process group

    if global_rank == 0:
        os.makedirs(os.path.join(args.output_dir, "y"), exist_ok=True)
        df = cudf.DataFrame(
            cupy.asarray(y).reshape((-1, 1)),
            columns=["y"],
            index=cupy.arange(num_nodes["paper"], dtype="int64"),
        )
        df.to_parquet(os.path.join(args.output_dir, "y", "y.parquet"))
