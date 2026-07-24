# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from cugraph.datasets import karate
from cugraph_pyg.utils.imports import import_optional, MissingModule

from cugraph_pyg.data import FeatureStore, GraphStore
from cugraph_pyg.loader import NeighborLoader

from pylibcugraph.comms import (
    cugraph_comms_create_unique_id,
    cugraph_comms_init,
    cugraph_comms_shutdown,
)

torch = import_optional("torch")
torch_geometric = import_optional("torch_geometric")
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


def run_test_graph_store_finalize(rank, uid, world_size):
    torch.cuda.set_device(rank)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    cugraph_comms_init(rank=rank, world_size=world_size, uid=uid, device=rank)

    df = karate.get_edgelist()
    src = torch.as_tensor(df["src"], device=f"cuda:{rank}").to(torch.int64)
    dst = torch.as_tensor(df["dst"], device=f"cuda:{rank}").to(torch.int64)
    edge_index = torch.tensor_split(torch.stack([dst, src]), world_size, dim=1)[rank]
    edge_type = ("person", "knows", "person")
    num_nodes = karate.number_of_nodes()

    graph_store = GraphStore()
    graph_store.put_edge_index(
        edge_index, edge_type, "coo", False, (num_nodes, num_nodes)
    )
    graph_store.finalize()

    with pytest.raises(NotImplementedError, match="Adding edges"):
        graph_store.put_edge_index(
            edge_index, edge_type, "coo", False, (num_nodes, num_nodes)
        )
    with pytest.raises(NotImplementedError, match="Removing edges"):
        graph_store.remove_edge_index(edge_type, "coo")

    feature_store = FeatureStore()
    feature_store["person", "feat", None] = torch.arange(num_nodes).reshape(-1, 1)
    input_nodes = torch.tensor_split(torch.arange(num_nodes), world_size)[rank]
    loader = NeighborLoader(
        (feature_store, graph_store),
        [5, 5],
        input_nodes=input_nodes,
        batch_size=num_nodes,
    )

    batch = next(iter(loader))
    assert isinstance(batch, torch_geometric.data.Data)
    assert (feature_store["person", "feat", None][batch.n_id] == batch.feat).all()

    cugraph_comms_shutdown()
    pylibwholegraph.torch.initialize.finalize()


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.mg
def test_graph_store_finalize_mg():
    uid = cugraph_comms_create_unique_id()
    world_size = torch.cuda.device_count()

    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    torch.multiprocessing.spawn(
        run_test_graph_store_finalize,
        args=(uid, world_size),
        nprocs=world_size,
    )
