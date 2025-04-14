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

# Single-node, multi-GPU example.

import argparse
import os
import time
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch.nn.parallel import DistributedDataParallel

import torch_geometric

# Ensures that a CUDA context is not created on import of rapids.
# Allows pytorch to create the context instead
os.environ["RAPIDS_NO_INITIALIZE"] = "1"


def init_pytorch_worker(rank, world_size, cugraph_id):
    import rmm

    rmm.reinitialize(
        devices=rank,
        managed_memory=True,
        pool_allocator=True,
    )

    import cupy

    cupy.cuda.Device(rank).use()
    from rmm.allocators.cupy import rmm_cupy_allocator

    cupy.cuda.set_allocator(rmm_cupy_allocator)

    torch.cuda.set_device(rank)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    from cugraph.gnn import cugraph_comms_init

    cugraph_comms_init(rank=rank, world_size=world_size, uid=cugraph_id, device=rank)


def run_train(
    rank,
    data,
    world_size,
    cugraph_id,
    model,
    epochs,
    batch_size,
    fan_out,
    split_idx,
    num_classes,
    wall_clock_start,
    num_layers=3,
    seeds_per_call=-1,
):

    init_pytorch_worker(
        rank,
        world_size,
        cugraph_id,
    )

    model = model.to(rank)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

    kwargs = dict(
        num_neighbors=[fan_out] * num_layers,
        batch_size=batch_size,
    )
    # Set Up Neighbor Loading
    from cugraph_pyg.data import GraphStore, TensorDictFeatureStore
    from cugraph_pyg.loader import NeighborLoader

    graph_store = GraphStore(is_multi_gpu=True)
    ixr = torch.tensor_split(data.edge_index, world_size, dim=1)[rank]
    graph_store[
        ("node", "rel", "node"), "coo", False, (data.num_nodes, data.num_nodes)
    ] = ixr

    feature_store = TensorDictFeatureStore()
    feature_store["node", "x", None] = data.x
    feature_store["node", "y", None] = data.y

    dist.barrier()

    ix_train = torch.tensor_split(split_idx["train"], world_size)[rank].cuda()
    train_loader = NeighborLoader(
        (feature_store, graph_store),
        input_nodes=ix_train,
        shuffle=True,
        drop_last=True,
        local_seeds_per_call=seeds_per_call if seeds_per_call > 0 else None,
        **kwargs,
    )

    ix_test = torch.tensor_split(split_idx["test"], world_size)[rank].cuda()
    test_loader = NeighborLoader(
        (feature_store, graph_store),
        input_nodes=ix_test,
        shuffle=True,
        drop_last=True,
        local_seeds_per_call=80000,
        **kwargs,
    )

    ix_valid = torch.tensor_split(split_idx["valid"], world_size)[rank].cuda()
    valid_loader = NeighborLoader(
        (feature_store, graph_store),
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

    if rank == 0:
        prep_time = round(time.perf_counter() - wall_clock_start, 2)
        print("Total time before training begins (prep_time) =", prep_time, "seconds")
        print("Beginning training...")
    for epoch in range(epochs):
        for i, batch in enumerate(train_loader):
            if i == warmup_steps:
                torch.cuda.synchronize()
                start = time.time()

            batch = batch.to(rank)
            batch_size = batch.batch_size

            batch.y = batch.y.to(torch.long)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss = F.cross_entropy(out[:batch_size], batch.y[:batch_size])
            loss.backward()
            optimizer.step()
            if rank == 0 and i % 10 == 0:
                print(
                    "Epoch: "
                    + str(epoch)
                    + ", Iteration: "
                    + str(i)
                    + ", Loss: "
                    + str(loss)
                )
        nb = i + 1.0

        if rank == 0:
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

                batch = batch.to(rank)
                batch_size = batch.batch_size

                batch.y = batch.y.to(torch.long)
                out = model(batch.x, batch.edge_index)[:batch_size]

                pred = out.argmax(dim=-1)
                y = batch.y[:batch_size].view(-1).to(torch.long)

                total_correct += int((pred == y).sum())
                total_examples += y.size(0)

            acc_val = total_correct / total_examples
            if rank == 0:
                print(
                    f"Validation Accuracy: {acc_val * 100.0:.4f}%",
                )

        torch.cuda.synchronize()

    with torch.no_grad():
        total_correct = total_examples = 0
        for i, batch in enumerate(test_loader):
            batch = batch.to(rank)
            batch_size = batch.batch_size

            batch.y = batch.y.to(torch.long)
            out = model(batch.x, batch.edge_index)[:batch_size]

            pred = out.argmax(dim=-1)
            y = batch.y[:batch_size].view(-1).to(torch.long)

            total_correct += int((pred == y).sum())
            total_examples += y.size(0)

        acc_test = total_correct / total_examples
        if rank == 0:
            print(
                f"Test Accuracy: {acc_test * 100.0:.4f}%",
            )

    if rank == 0:
        total_time = round(time.perf_counter() - wall_clock_start, 2)
        print("Total Program Runtime (total_time) =", total_time, "seconds")
        print("total_time - prep_time =", total_time - prep_time, "seconds")

    from cugraph.gnn import cugraph_comms_shutdown

    cugraph_comms_shutdown()
    dist.destroy_process_group()


if __name__ == "__main__":
    if os.getenv("CI", "false").lower() == "true":
        warnings.warn("Skipping SMNG example in CI due to memory limit")
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--hidden_channels", type=int, default=256)
        parser.add_argument("--num_layers", type=int, default=2)
        parser.add_argument("--lr", type=float, default=0.001)
        parser.add_argument("--epochs", type=int, default=4)
        parser.add_argument("--batch_size", type=int, default=1024)
        parser.add_argument("--fan_out", type=int, default=30)
        parser.add_argument("--dataset_root", type=str, default="datasets")
        parser.add_argument("--dataset", type=str, default="ogbn-products")
        parser.add_argument("--seeds_per_call", type=int, default=-1)

        parser.add_argument(
            "--n_devices",
            type=int,
            default=-1,
            help="1-8 to use that many GPUs. Defaults to all available GPUs",
        )

        args = parser.parse_args()
        wall_clock_start = time.perf_counter()

        from rmm.allocators.torch import rmm_torch_allocator

        torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

        with torch.serialization.safe_globals(
            [
                torch_geometric.data.data.DataEdgeAttr,
                torch_geometric.data.data.DataTensorAttr,
                torch_geometric.data.storage.GlobalStorage,
            ]
        ):
            dataset = PygNodePropPredDataset(name=args.dataset, root=args.dataset_root)
            split_idx = dataset.get_idx_split()
            data = dataset[0]
            data.y = data.y.reshape(-1)

        model = torch_geometric.nn.models.GCN(
            dataset.num_features,
            args.hidden_channels,
            args.num_layers,
            dataset.num_classes,
        )

        print("Data =", data)
        if args.n_devices == -1:
            world_size = torch.cuda.device_count()
        else:
            world_size = args.n_devices
        print("Using", world_size, "GPUs...")

        # Create the uid needed for cuGraph comms
        from cugraph.gnn import cugraph_comms_create_unique_id

        cugraph_id = cugraph_comms_create_unique_id()

        mp.spawn(
            run_train,
            args=(
                data,
                world_size,
                cugraph_id,
                model,
                args.epochs,
                args.batch_size,
                args.fan_out,
                split_idx,
                dataset.num_classes,
                wall_clock_start,
                args.num_layers,
                args.seeds_per_call,
            ),
            nprocs=world_size,
            join=True,
        )
