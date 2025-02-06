# Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

import pytest

import dgl
import torch
import numpy as np

import cugraph_dgl

from cugraph.testing.mg_utils import (
    start_dask_client,
    stop_dask_client,
)

from cugraph.datasets import karate


@pytest.fixture(scope="module")
def dask_client():
    # start_dask_client will check for the SCHEDULER_FILE and
    # DASK_WORKER_DEVICES env vars and use them when creating a client if
    # set. start_dask_client will also initialize the Comms singleton.
    dask_client, dask_cluster = start_dask_client(
        dask_worker_devices="0", protocol="tcp"
    )

    yield dask_client

    stop_dask_client(dask_client, dask_cluster)


class SparseGraphData1:
    size = (6, 5)
    nnz = 6
    src_ids = torch.IntTensor([0, 1, 2, 3, 2, 5]).cuda()
    dst_ids = torch.IntTensor([1, 2, 3, 4, 0, 3]).cuda()
    values = torch.IntTensor([10, 20, 30, 40, 50, 60]).cuda()

    # CSR
    src_ids_sorted_by_src = torch.IntTensor([0, 1, 2, 2, 3, 5]).cuda()
    dst_ids_sorted_by_src = torch.IntTensor([1, 2, 0, 3, 4, 3]).cuda()
    csrc_ids = torch.IntTensor([0, 1, 2, 4, 5, 5, 6]).cuda()
    values_csr = torch.IntTensor([10, 20, 50, 30, 40, 60]).cuda()

    # CSC
    src_ids_sorted_by_dst = torch.IntTensor([2, 0, 1, 5, 2, 3]).cuda()
    dst_ids_sorted_by_dst = torch.IntTensor([0, 1, 2, 3, 3, 4]).cuda()
    cdst_ids = torch.IntTensor([0, 1, 2, 3, 5, 6]).cuda()
    values_csc = torch.IntTensor([50, 10, 20, 60, 30, 40]).cuda()


@pytest.fixture
def sparse_graph_1():
    return SparseGraphData1()


@pytest.fixture
def dgl_graph_1():
    src = torch.tensor([0, 1, 0, 2, 3, 0, 4, 0, 5, 0, 6, 7, 0, 8, 9])
    dst = torch.tensor([1, 9, 2, 9, 9, 4, 9, 5, 9, 6, 9, 9, 8, 9, 0])
    return dgl.graph((src, dst))


def create_karate_bipartite(multi_gpu: bool = False):
    df = karate.get_edgelist()
    df.src = df.src.astype("int64")
    df.dst = df.dst.astype("int64")

    graph = cugraph_dgl.Graph(is_multi_gpu=multi_gpu)
    total_num_nodes = max(df.src.max(), df.dst.max()) + 1

    num_nodes_group_1 = total_num_nodes // 2
    num_nodes_group_2 = total_num_nodes - num_nodes_group_1

    node_x_1 = np.random.random((num_nodes_group_1,))
    node_x_2 = np.random.random((num_nodes_group_2,))

    if multi_gpu:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        node_x_1 = np.array_split(node_x_1, world_size)[rank]
        node_x_2 = np.array_split(node_x_2, world_size)[rank]

    graph.add_nodes(num_nodes_group_1, {"x": node_x_1}, "type1")
    graph.add_nodes(num_nodes_group_2, {"x": node_x_2}, "type2")

    edges = {}
    edges["type1", "e1", "type1"] = df[
        (df.src < num_nodes_group_1) & (df.dst < num_nodes_group_1)
    ]
    edges["type1", "e2", "type2"] = df[
        (df.src < num_nodes_group_1) & (df.dst >= num_nodes_group_1)
    ]
    edges["type2", "e3", "type1"] = df[
        (df.src >= num_nodes_group_1) & (df.dst < num_nodes_group_1)
    ]
    edges["type2", "e4", "type2"] = df[
        (df.src >= num_nodes_group_1) & (df.dst >= num_nodes_group_1)
    ]

    edges["type1", "e2", "type2"].dst -= num_nodes_group_1
    edges["type2", "e3", "type1"].src -= num_nodes_group_1
    edges["type2", "e4", "type2"].dst -= num_nodes_group_1
    edges["type2", "e4", "type2"].src -= num_nodes_group_1

    if multi_gpu:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        edges_local = {
            etype: edf.iloc[np.array_split(np.arange(len(edf)), world_size)[rank]]
            for etype, edf in edges.items()
        }
    else:
        edges_local = edges

    for etype, edf in edges_local.items():
        graph.add_edges(edf.src, edf.dst, etype=etype)

    return graph, edges, (num_nodes_group_1, num_nodes_group_2)


@pytest.fixture
def karate_bipartite():
    return create_karate_bipartite(False)
