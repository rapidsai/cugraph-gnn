#!/usr/bin/env python3

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

# Multi-node, multi-GPU example with WholeGraph feature storage.
# Can be run with torchrun.

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel

from torch_geometric.nn.models import GCN

from pylibwholegraph.torch.initialize import (
    init as wm_init,
    finalize as wm_finalize,
)

# Allow computation on objects that are larger than GPU memory
os.environ["CUDF_SPILL"] = "1"
# Allows pytorch to create the CUDA context
os.environ["RAPIDS_NO_INITIALIZE"] = "1"


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


def load_and_distribute_data(rank):
    from cugraph_pyg.data import GraphStore, WholeFeatureStore

    # Only rank 0 loads the dataset
    if rank == 0:
        dataset = PygNodePropPredDataset(name="ogbn-papers100M")
        split_idx = dataset.get_idx_split()
        data = dataset[0]

        # Create metadata
        meta = {
            "num_classes": int(dataset.num_classes),
            "num_features": int(dataset.num_features),
            "num_nodes": int(data.num_nodes),
        }
    else:
        data = None
        split_idx = None
        meta = None

    # Broadcast metadata to all ranks
    if rank == 0:
        meta_tensor = torch.tensor(
            [meta["num_classes"], meta["num_features"], meta["num_nodes"]],
            dtype=torch.long,
        )
    else:
        meta_tensor = torch.zeros(3, dtype=torch.long)
    dist.broadcast(meta_tensor, src=0)

    if rank != 0:
        meta = {
            "num_classes": int(meta_tensor[0]),
            "num_features": int(meta_tensor[1]),
            "num_nodes": int(meta_tensor[2]),
        }

    # Initialize stores
    graph_store = GraphStore(is_multi_gpu=True)
    feature_store = WholeFeatureStore()

    # Distribute data using WholeFeatureStore
    if rank == 0:
        feature_store["node", "x", None] = data.x
        feature_store["node", "y", None] = data.y
        graph_store[
            ("node", "rel", "node"),
            "coo",
            False,
            (meta["num_nodes"], meta["num_nodes"]),
        ] = data.edge_index

    # Synchronize to ensure data is distributed
    dist.barrier()

    # Distribute split indices
    if rank == 0:
        for split in ["train", "test", "valid"]:
            split_size = split_idx[split].numel()
            dist.broadcast(torch.tensor([split_size], dtype=torch.long), src=0)
            dist.broadcast(split_idx[split], src=0)
    else:
        split_idx = {}
        for split in ["train", "test", "valid"]:
            size_tensor = torch.zeros(1, dtype=torch.long)
            dist.broadcast(size_tensor, src=0)
            split_idx[split] = torch.zeros(size_tensor[0], dtype=torch.long)
            dist.broadcast(split_idx[split], src=0)

    return (feature_store, graph_store), split_idx, meta


def run_train(
    global_rank,
    data,
    split_idx,
    world_size,
    device,
    model,
    epochs,
    batch_size,
    fan_out,
    num_classes,
    wall_clock_start,
    num_layers=3,
    in_memory=False,
    seeds_per_call=-1,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

    kwargs = dict(
        num_neighbors=[fan_out] * num_layers,
        batch_size=batch_size,
    )

    from cugraph_pyg.loader import NeighborLoader

    ix_train = split_idx["train"].cuda()
    train_loader = NeighborLoader(
        data,
        input_nodes=ix_train,
        shuffle=True,
        drop_last=True,
        local_seeds_per_call=seeds_per_call if seeds_per_call > 0 else None,
        **kwargs,
    )

    ix_test = split_idx["test"].cuda()
    test_loader = NeighborLoader(
        data,
        input_nodes=ix_test,
        shuffle=True,
        drop_last=True,
        local_seeds_per_call=80000,
        **kwargs,
    )

    ix_valid = split_idx["valid"].cuda()
    valid_loader = NeighborLoader(
        data,
        input_nodes=ix_valid,
        shuffle=True,
        drop_last=True,
        local_seeds_per_call=seeds_per_call if seeds_per_call > 0 else None,
        **kwargs,
    )

    dist.barrier()

    eval_steps = 1000
    warmup_steps = 20
    dist.barrier()
    torch.cuda.synchronize()

    if global_rank == 0:
        prep_time = round(time.perf_counter() - wall_clock_start, 2)
        print("Total time before training begins (prep_time) =", prep_time, "seconds")
        print("Beginning training...")

    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            if i == warmup_steps:
                torch.cuda.synchronize()
                start = time.time()

            batch = batch.to(device)
            batch_size = batch.batch_size

            batch.y = batch.y.view(-1).to(torch.long)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out[:batch_size], batch.y[:batch_size])
            loss.backward()
            optimizer.step()

            if global_rank == 0 and i % 10 == 0:
                print(
                    "Epoch: "
                    + str(epoch)
                    + ", Iteration: "
                    + str(i)
                    + ", Loss: "
                    + str(loss)
                )
        nb = i + 1.0

        if global_rank == 0:
            print(
                "Average Training Iteration Time:",
                (time.time() - start) / (nb - warmup_steps),
                "s/iter",
            )

        with torch.no_grad():
            total_correct = total_examples = 0
            for i, batch in enumerate(valid_loader):
                if i >= eval_steps:
                    break

                batch = batch.to(device)
                batch_size = batch.batch_size

                batch.y = batch.y.to(torch.long)
                out = model(batch.x, batch.edge_index)[:batch_size]

                pred = out.argmax(dim=-1)
                y = batch.y[:batch_size].view(-1).to(torch.long)

                total_correct += int((pred == y).sum())
                total_examples += y.size(0)

            acc_val = total_correct / total_examples
            if global_rank == 0:
                print(
                    f"Validation Accuracy: {acc_val * 100.0:.4f}%",
                )

        torch.cuda.synchronize()

    with torch.no_grad():
        total_correct = total_examples = 0
        for i, batch in enumerate(test_loader):
            batch = batch.to(device)
            batch_size = batch.batch_size

            batch.y = batch.y.to(torch.long)
            out = model(batch.x, batch.edge_index)[:batch_size]

            pred = out.argmax(dim=-1)
            y = batch.y[:batch_size].view(-1).to(torch.long)

            total_correct += int((pred == y).sum())
            total_examples += y.size(0)

        acc_test = total_correct / total_examples
        if global_rank == 0:
            print(
                f"Test Accuracy: {acc_test * 100.0:.4f}%",
            )

    if global_rank == 0:
        total_time = round(time.perf_counter() - wall_clock_start, 2)
        print("Total Program Runtime (total_time) =", total_time, "seconds")
        print("total_time - prep_time =", total_time - prep_time, "seconds")

    wm_finalize()

    from cugraph.gnn import cugraph_comms_shutdown

    cugraph_comms_shutdown()


def parse_args():
    parser = argparse.ArgumentParser(description="GCN Training with WholeGraph")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--fan_out", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--seeds_per_call", type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Initialize process group
    dist.init_process_group("nccl")
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    # Initialize worker
    init_pytorch_worker(global_rank, local_rank, world_size, 0)

    wall_clock_start = time.perf_counter()

    # Load and distribute data
    data, split_idx, meta = load_and_distribute_data(global_rank)

    device = torch.device(f"cuda:{local_rank}")
    model = GCN(
        meta["num_features"],
        args.hidden_channels,
        args.num_layers,
        meta["num_classes"],
    ).to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank])

    # Train
    run_train(
        global_rank=global_rank,
        data=data,
        split_idx=split_idx,
        world_size=world_size,
        device=device,
        model=model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        fan_out=args.fan_out,
        num_classes=meta["num_classes"],
        wall_clock_start=wall_clock_start,
        num_layers=args.num_layers,
        seeds_per_call=args.seeds_per_call,
    )

    # Cleanup
    wm_finalize()
    dist.destroy_process_group()
