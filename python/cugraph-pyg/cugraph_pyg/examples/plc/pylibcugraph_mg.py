# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# This example shows how to use cuGraph nccl-only comms, pylibcuGraph,
# and PyTorch DDP to run a multi-GPU workflow.  Most users of the
# GNN packages will not interact with cuGraph directly.  This example
# is intented for users who want to extend cuGraph within a DDP workflow.

import os
import argparse

import pandas
import numpy as np
import torch
import torch.multiprocessing as tmp
import torch.distributed as dist

import cudf

from pylibcugraph.comms import (
    cugraph_comms_init,
    cugraph_comms_shutdown,
    cugraph_comms_create_unique_id,
    cugraph_comms_get_raft_handle,
)

from pylibcugraph import MGGraph, ResourceHandle, GraphProperties, degrees

from ogb.nodeproppred import NodePropPredDataset


def init_pytorch(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def calc_degree(rank: int, world_size: int, uid, edgelist):
    init_pytorch(rank, world_size)

    device = rank
    cugraph_comms_init(rank, world_size, uid, device)

    print(f"rank {rank} initialized cugraph")

    src = cudf.Series(np.array_split(edgelist[0], world_size)[rank])
    dst = cudf.Series(np.array_split(edgelist[1], world_size)[rank])

    seeds = cudf.Series(np.arange(rank * 50, (rank + 1) * 50))
    handle = ResourceHandle(cugraph_comms_get_raft_handle().getHandle())

    print("constructing graph")
    G = MGGraph(
        handle,
        GraphProperties(is_multigraph=True, is_symmetric=False),
        [src],
        [dst],
    )
    print("graph constructed")

    print("calculating degrees")
    vertices, in_deg, out_deg = degrees(handle, G, seeds, do_expensive_check=False)
    print("degrees calculated")

    print("constructing dataframe")
    df = pandas.DataFrame(
        {"v": vertices.get(), "in": in_deg.get(), "out": out_deg.get()}
    )
    print(df)

    dist.barrier()
    cugraph_comms_shutdown()
    print(f"rank {rank} shut down cugraph")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="datasets",
        help="Root directory for dataset storage",
    )
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    uid = cugraph_comms_create_unique_id()

    dataset = NodePropPredDataset("ogbn-products", root=args.dataset_root)
    el = dataset[0][0]["edge_index"].astype("int64")

    tmp.spawn(
        calc_degree,
        args=(world_size, uid, el),
        nprocs=world_size,
    )


if __name__ == "__main__":
    main()
