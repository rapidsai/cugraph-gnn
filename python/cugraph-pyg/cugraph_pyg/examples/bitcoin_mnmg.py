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

import os
import argparse

import torch

from torch_geometric.datasets import EllipticBitcoinDataset

from torch_geometric.nn.models import GraphSAGE, GCN, GAT

import torch.nn.functional as F


def create_uid(global_rank, device):
    # Create the uid needed for cuGraph comms
    if global_rank == 0:
        from cugraph.gnn import cugraph_comms_create_unique_id

        cugraph_id = [cugraph_comms_create_unique_id()]
    else:
        cugraph_id = [None]

    torch.distributed.broadcast_object_list(cugraph_id, src=0, device=device)

    cugraph_id = cugraph_id[0]
    return cugraph_id


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

    torch.cuda.set_device(local_rank)

    from cugraph.gnn import cugraph_comms_init

    cugraph_comms_init(
        rank=global_rank, world_size=world_size, uid=cugraph_id, device=local_rank
    )

    # WholeGraph is already initialized.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="./data/")
    parser.add_argument("--encoder", type=str, default="sage")
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./results")
    args = parser.parse_args()

    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    cugraph_id = create_uid(rank, device=torch.device(f"cuda:{local_rank}"))
    init_pytorch_worker(rank, local_rank, world_size, cugraph_id)

    dataset = EllipticBitcoinDataset(root=args.dataset_root)
    data = dataset[0]
    assert dataset.num_classes == 2

    from cugraph_pyg.data import GraphStore, FeatureStore
    from cugraph_pyg.tensor import empty

    graph_store = GraphStore(is_multi_gpu=True)
    feature_store = FeatureStore()

    # Distribute data (will evenly distribute from rank 0;
    # all other ranks pass empties and receive their slice)
    graph_store[
        ("entity", "transaction", "entity"),
        "coo",
        False,
        (data.num_nodes, data.num_nodes),
    ] = (
        data.edge_index if rank == 0 else empty(dim=2)
    )
    feature_store["entity", "x", None] = data.x if rank == 0 else empty(dim=2)
    feature_store["entity", "y", None] = data.y if rank == 0 else empty(dim=1)
    torch.distributed.barrier()

    if args.encoder.lower() == "sage":
        encoder = GraphSAGE(
            in_channels=data.x.shape[1],
            hidden_channels=args.hidden_channels,
            out_channels=dataset.num_classes,
            num_layers=2,
        )
    elif args.encoder.lower() == "gcn":
        encoder = GCN(
            in_channels=data.x.shape[1],
            hidden_channels=args.hidden_channels,
            out_channels=dataset.num_classes,
            num_layers=2,
        )
    elif args.encoder.lower() == "gat":
        encoder = GAT(
            in_channels=data.x.shape[1],
            hidden_channels=args.hidden_channels,
            out_channels=dataset.num_classes,
            num_layers=2,
        )
    else:
        raise ValueError(f"Invalid encoder: {args.encoder}")

    encoder = torch.nn.parallel.DistributedDataParallel(
        encoder.cuda(), device_ids=[local_rank]
    )
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    ix_train = torch.tensor_split(
        torch.arange(data.num_nodes, device="cuda")[data.train_mask], world_size
    )[rank]

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_neighbors": [25, 10],
        "shuffle": True,
        "drop_last": True,
    }

    from cugraph_pyg.loader import NeighborLoader

    train_loader = NeighborLoader(
        (feature_store, graph_store),
        input_nodes=ix_train,
        **loader_kwargs,
    )

    test_loader = NeighborLoader(
        (feature_store, graph_store),
        input_nodes=ix_train,
        **loader_kwargs,
    )

    for epoch in range(1, args.num_epochs + 1):
        for it, batch in enumerate(train_loader):
            optimizer.zero_grad()
            out = encoder(batch.x, batch.edge_index)

            # only the initial batch is labeled
            loss = F.cross_entropy(out[: batch.batch_size], batch.y[: batch.batch_size])
            loss.backward()
            optimizer.step()

            if rank == 0 and it % 10 == 0:
                print(f"Epoch {epoch} iter {it} loss: {loss.item()}")

    with torch.no_grad():
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        for batch in test_loader:
            out = encoder(batch.x, batch.edge_index)

            loss = F.cross_entropy(out[: batch.batch_size], batch.y[: batch.batch_size])
            total_loss += loss.item() * batch.batch_size
            total_examples += batch.batch_size
            total_correct += (
                (out[: batch.batch_size].argmax(dim=-1) == batch.y[: batch.batch_size])
                .sum()
                .item()
            )
        print(
            f"rank={rank} Test loss: {total_loss / total_examples}"
            f" acc: {total_correct / total_examples}"
        )

    torch.distributed.barrier()

    inf_loader = NeighborLoader(
        (feature_store, graph_store),
        input_nodes=ix_train,
        num_neighbors=[-1],
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    feature_store["entity", "emb", None] = (
        torch.zeros(
            (data.num_nodes, args.hidden_channels), dtype=torch.float32, device="cuda"
        )
        if rank == 0
        else empty(dim=2)
    )

    with torch.no_grad():
        total_correct = total_examples = 0
        for batch in inf_loader:
            x = batch.x
            for layer in range(encoder.module.num_layers - 1):
                x = encoder.module.inference_per_layer(
                    layer, x, batch.edge_index, batch.batch_size
                )
            x = x[: batch.batch_size]
            feature_store["entity", "emb", None][batch.n_id[: batch.batch_size]] = x

    import cudf

    df = cudf.DataFrame(
        feature_store["entity", "emb", None].get_local_tensor(),
        index=None,
        columns=[f"emb_{i}" for i in range(args.hidden_channels)],
    )
    df["y"] = (feature_store["entity", "y", None]).get_local_tensor()

    df.to_parquet(
        f"{args.output_dir}/emb_{args.encoder}_{args.hidden_channels}"
        f"_{args.batch_size}_{args.lr}_{args.num_epochs}_{rank}.parquet"
    )
