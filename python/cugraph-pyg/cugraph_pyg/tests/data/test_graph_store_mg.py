# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from cugraph.datasets import karate
from cugraph_pyg.utils.imports import import_optional, MissingModule

from cugraph_pyg.data import GraphStore

torch = import_optional("torch")
pylibwholegraph = import_optional("pylibwholegraph")


def run_test_graph_store_basic_api(rank, world_size):
    torch.cuda.set_device(rank)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    # Load and distribute the graph data
    df = karate.get_edgelist()
    src = torch.as_tensor(df["src"], device=f"cuda:{rank}").to(torch.int64)
    dst = torch.as_tensor(df["dst"], device=f"cuda:{rank}").to(torch.int64)

    # Split the edge indices across GPUs
    num_edges = len(src)
    edges_per_gpu = num_edges // world_size
    start_idx = rank * edges_per_gpu
    end_idx = start_idx + edges_per_gpu if rank < world_size - 1 else num_edges

    local_src = src[start_idx:end_idx]
    local_dst = dst[start_idx:end_idx]

    ei = torch.stack([local_dst, local_src])
    num_nodes = karate.number_of_nodes()

    # Create and populate the graph store
    graph_store = GraphStore()
    graph_store.put_edge_index(
        ei, ("person", "knows", "person"), "coo", False, (num_nodes, num_nodes)
    )

    # Verify the edge indices
    rei = graph_store.get_edge_index(("person", "knows", "person"), "coo")

    # All gather the edge indices from all ranks
    # First, gather the sizes from each rank
    local_size = torch.tensor([rei.shape[1]], device=f"cuda:{rank}")
    gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_sizes, local_size)

    # Now gather the edge indices with properly sized buffers
    gathered_rei = [
        torch.zeros((2, size.item()), dtype=rei.dtype, device=f"cuda:{rank}")
        for size in gathered_sizes
    ]
    torch.distributed.all_gather(gathered_rei, rei)
    gathered_rei = torch.concat(gathered_rei, dim=1)

    assert (gathered_rei == torch.stack([dst, src])).all()

    # Verify edge attributes
    edge_attrs = graph_store.get_all_edge_attrs()
    assert len(edge_attrs) == 1

    # Test removal
    graph_store.remove_edge_index(("person", "knows", "person"), "coo")
    edge_attrs = graph_store.get_all_edge_attrs()
    assert len(edge_attrs) == 0

    pylibwholegraph.torch.initialize.finalize()


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.mg
def test_graph_store_basic_api_mg():
    world_size = torch.cuda.device_count()

    # simulate torchrun call
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    torch.multiprocessing.spawn(
        run_test_graph_store_basic_api,
        args=(world_size,),
        nprocs=world_size,
    )
