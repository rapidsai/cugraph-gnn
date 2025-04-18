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

import os

import pytest

from cugraph.datasets import karate
from cugraph.utilities.utils import import_optional, MissingModule

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
    src = torch.as_tensor(df["src"], device=f"cuda:{rank}")
    dst = torch.as_tensor(df["dst"], device=f"cuda:{rank}")

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
    graph_store = GraphStore(is_multi_gpu=True)
    graph_store.put_edge_index(
        ei, ("person", "knows", "person"), "coo", False, (num_nodes, num_nodes)
    )

    # Verify the edge indices
    rei = graph_store.get_edge_index(("person", "knows", "person"), "coo")
    assert (local_dst == rei[0]).all()
    assert (local_src == rei[1]).all()

    # Verify edge attributes
    edge_attrs = graph_store.get_all_edge_attrs()
    assert len(edge_attrs) == 1

    # Test removal
    graph_store.remove_edge_index(("person", "knows", "person"), "coo")
    edge_attrs = graph_store.get_all_edge_attrs()
    assert len(edge_attrs) == 0


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
