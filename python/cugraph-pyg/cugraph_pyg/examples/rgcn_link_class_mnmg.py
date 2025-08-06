# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

# This example illustrates link classification using the ogbl-wikikg2 dataset.

import os
import argparse
import warnings

import numpy

import torch
import torch_geometric

import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import FastRGCNConv, GAE
from torch.nn.parallel import DistributedDataParallel

from ogb.linkproppred import PygLinkPropPredDataset

from pylibwholegraph.torch.initialize import (
    init as wm_init,
    finalize as wm_finalize,
)

# Ensures that a CUDA context is not created on import of rapids.
# Allows pytorch to create the context instead
os.environ["RAPIDS_NO_INITIALIZE"] = "1"


def init_pytorch_worker(global_rank, local_rank, world_size, uid):
    import rmm

    rmm.reinitialize(devices=[local_rank], pool_allocator=False, managed_memory=True)

    import cupy
    from rmm.allocators.cupy import rmm_cupy_allocator

    cupy.cuda.Device(local_rank).use()
    cupy.cuda.set_allocator(rmm_cupy_allocator)

    torch.cuda.set_device(local_rank)

    from cugraph.gnn import cugraph_comms_init

    cugraph_comms_init(
        global_rank,
        world_size,
        uid,
        local_rank,
    )

    wm_init(global_rank, world_size, local_rank, torch.cuda.device_count())


class RGCNEncoder(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels, num_relations, num_bases=30):
        super().__init__()
        self.node_emb = Parameter(torch.empty(num_nodes, hidden_channels))
        self.conv1 = FastRGCNConv(
            hidden_channels, hidden_channels, num_relations, num_bases=num_bases
        )
        self.conv2 = FastRGCNConv(
            hidden_channels, hidden_channels, num_relations, num_bases=num_bases
        )
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, edge_index, edge_type):
        x = self.node_emb
        x = self.conv1(x, edge_index, edge_type).relu_()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x


def get_local_split(t):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    u = t()
    return u[torch.tensor_split(torch.arange(u.shape[0]), world_size, dim=0)[rank]]


def train(epoch, model, optimizer, train_loader, edge_feature_store, num_steps=None):
    model.train()
    optimizer.zero_grad()

    for i, batch in enumerate(train_loader):
        r = (
            edge_feature_store[("n", "e", "n"), "rel", None][batch.e_id]
            .flatten()
            .cuda()
        ).to(torch.int64)
        z = model.encode(batch.edge_index, r)

        loss = model.recon_loss(z, batch.edge_index)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(
                f"Epoch: {epoch:02d}, Iteration: {i:02d}, Loss: {loss:.4f}", flush=True
            )
        if num_steps and i == num_steps:
            break


def test(stage, epoch, model, loader, num_steps=None):
    # TODO support ROC-AUC metric
    # Predict probabilities of future edges
    model.eval()

    rr = 0.0
    for i, (h, h_neg, t, t_neg, r) in enumerate(loader):
        if num_steps and i >= num_steps:
            break

        ei = torch.concatenate(
            [
                torch.stack([h, t]).cuda(),
                torch.stack([h_neg.flatten(), t_neg.flatten()]).cuda(),
            ],
            dim=-1,
        )

        r = (
            torch.concatenate([r, torch.repeat_interleave(r, h_neg.shape[-1])])
            .cuda()
            .to(torch.int64)
        )

        z = model.encode(ei, r)
        q = model.decode(z, ei)

        _, ix = torch.sort(q, descending=True)
        rr += 1.0 / (1.0 + ix[0])

    print(f"epoch {epoch:02d} {stage} mrr:", rr / i, flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--num_neg", type=int, default=500)
    parser.add_argument("--num_pos", type=int, default=-1)
    parser.add_argument("--fan_out", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="ogbl-wikikg2")
    parser.add_argument("--dataset_root", type=str, default="datasets")
    parser.add_argument("--seeds_per_call", type=int, default=-1)
    parser.add_argument("--n_devices", type=int, default=-1)

    return parser.parse_args()


def run_train(global_rank, local_rank, model, data, edge_feature_store, splits, args):
    model = model.to(torch.device(local_rank))
    model = GAE(DistributedDataParallel(model, device_ids=[local_rank]))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    eli = torch.stack(
        [
            get_local_split(splits["train"]["head"]).cpu(),
            get_local_split(splits["train"]["tail"]).cpu(),
        ]
    )

    from cugraph_pyg.loader import LinkNeighborLoader

    print("creating train loader...", flush=True)
    train_loader = LinkNeighborLoader(
        data,
        [args.fan_out] * args.num_layers,
        edge_label_index=eli,
        local_seeds_per_call=args.seeds_per_call if args.seeds_per_call > 0 else None,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    def get_eval_loader(stage: str):
        head = get_local_split(splits[stage]["head"]).cpu()
        tail = get_local_split(splits[stage]["tail"]).cpu()

        head_neg = get_local_split(splits[stage]["head_neg"])[:, : args.num_neg].cpu()
        tail_neg = get_local_split(splits[stage]["tail_neg"])[:, : args.num_neg].cpu()

        rel = get_local_split(splits[stage]["relation"]).cpu()

        print(
            head.shape,
            head_neg.shape,
            tail.shape,
            tail_neg.shape,
            rel.shape,
            flush=True,
        )

        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                head,
                head_neg,
                tail,
                tail_neg,
                rel,
            ),
            batch_size=1,
            shuffle=False,
            drop_last=True,
        )

    test_loader = get_eval_loader("test")
    valid_loader = get_eval_loader("valid")

    num_train_steps = (args.num_pos // args.batch_size) if args.num_pos > 0 else 100

    for epoch in range(1, 1 + args.epochs):
        train(
            epoch,
            model,
            optimizer,
            train_loader,
            edge_feature_store,
            num_steps=num_train_steps,
        )
        test("validation", epoch, model, valid_loader, num_steps=1024)

    test("test", epoch, model, test_loader, num_steps=1024)

    wm_finalize()

    from cugraph.gnn import cugraph_comms_shutdown

    cugraph_comms_shutdown()


if __name__ == "__main__":
    args = parse_args()

    if "LOCAL_RANK" in os.environ:
        torch.distributed.init_process_group("nccl")
        world_size = torch.distributed.get_world_size()
        global_rank = torch.distributed.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(local_rank)

        # Create the uid needed for cuGraph comms
        if global_rank == 0:
            from cugraph.gnn import cugraph_comms_create_unique_id

            cugraph_id = [cugraph_comms_create_unique_id()]
        else:
            cugraph_id = [None]
        torch.distributed.broadcast_object_list(cugraph_id, src=0, device=device)
        cugraph_id = cugraph_id[0]

        init_pytorch_worker(global_rank, local_rank, world_size, cugraph_id)

        torch.distributed.barrier()
        from rmm.allocators.torch import rmm_torch_allocator

        with torch.cuda.use_mem_pool(
            torch.cuda.MemPool(rmm_torch_allocator.allocator())
        ):

            if global_rank == 0:
                with torch.serialization.safe_globals(
                    [
                        torch_geometric.data.data.DataEdgeAttr,
                        torch_geometric.data.data.DataTensorAttr,
                        torch_geometric.data.storage.GlobalStorage,
                        numpy.core.multiarray._reconstruct,
                        numpy.ndarray,
                        numpy.dtype,
                        numpy.dtypes.Int64DType,
                    ]
                ):
                    data = PygLinkPropPredDataset(args.dataset, root=args.dataset_root)
                    dataset = data[0]
                    print(dataset, flush=True)

                    splits = data.get_edge_split()

                nr = [dataset.num_nodes, int(dataset.edge_reltype.max()) + 1]
            else:
                nr = [0, 0]

            torch.distributed.barrier()
            torch.distributed.broadcast_object_list(nr, src=0, device=device)
            num_nodes, num_rels = nr

            print(
                f"num_nodes: {num_nodes}, num_rels: {num_rels}, rank: {global_rank}",
                flush=True,
            )
            torch.distributed.barrier()

            from cugraph_pyg.data import FeatureStore, GraphStore
            from cugraph_pyg.tensor import empty

            edge_feature_store = FeatureStore()
            splits_storage = FeatureStore()
            feature_store = torch_geometric.data.HeteroData()
            graph_store = GraphStore()
            torch.distributed.barrier()

            print(f"broadcasting edge rel type (rank {global_rank})", flush=True)
            edge_feature_store[("n", "e", "n"), "rel", None] = (
                dataset.edge_reltype.to(torch.int32)
                if global_rank == 0
                else empty(dim=2)
            )

            print(f"broadcasting edge index (rank {global_rank})", flush=True)
            graph_store[("n", "e", "n"), "coo", False, (num_nodes, num_nodes)] = (
                dataset.edge_index if global_rank == 0 else empty(dim=2)
            )

            print("broadcasting splits", flush=True)
            for stage in ["train", "test", "valid"]:
                splits_storage[stage, "head", None] = (
                    splits[stage]["head"].to(torch.int64)
                    if global_rank == 0
                    else empty(dim=1)
                )
                splits_storage[stage, "tail", None] = (
                    splits[stage]["tail"].to(torch.int64)
                    if global_rank == 0
                    else empty(dim=1)
                )

            print("broadcasting negative splits", flush=True)
            for stage in ["test", "valid"]:
                splits_storage[stage, "head_neg", None] = (
                    splits[stage]["head_neg"].to(torch.int64)
                    if global_rank == 0
                    else empty(dim=2)
                )
                splits_storage[stage, "tail_neg", None] = (
                    splits[stage]["tail_neg"].to(torch.int64)
                    if global_rank == 0
                    else empty(dim=2)
                )
                splits_storage[stage, "relation", None] = (
                    splits[stage]["relation"].to(torch.int32)
                    if global_rank == 0
                    else empty(dim=1)
                )

            print("reached barrier", flush=True)
            torch.distributed.barrier()

            model = RGCNEncoder(
                num_nodes,
                hidden_channels=args.hidden_channels,
                num_relations=num_rels,
            )

            if global_rank == 0:
                del data

            run_train(
                global_rank,
                local_rank,
                model,
                (feature_store, graph_store),
                edge_feature_store,
                splits_storage,
                args,
            )
    else:
        warnings.warn("This script should be run with 'torchrun`.  Exiting.")
