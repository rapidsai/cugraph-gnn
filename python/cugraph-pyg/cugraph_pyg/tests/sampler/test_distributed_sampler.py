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

import pytest

import cupy

from cugraph_pyg.sampler import UniformNeighborSampler

from pylibcugraph import SGGraph, ResourceHandle, GraphProperties

from cugraph.utilities.utils import import_optional, MissingModule


torch = import_optional("torch")


@pytest.mark.sg
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
def test_dist_sampler_hetero_from_nodes():
    props = GraphProperties(
        is_symmetric=False,
        is_multigraph=True,
    )

    handle = ResourceHandle()

    srcs = cupy.array([4, 5, 6, 7, 8, 9, 8, 9, 8, 7, 6, 5, 4, 5])
    dsts = cupy.array([0, 1, 2, 3, 3, 0, 4, 5, 6, 8, 7, 8, 9, 9])
    eids = cupy.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7])
    etps = cupy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], dtype="int32")

    graph = SGGraph(
        handle,
        props,
        srcs,
        dsts,
        vertices_array=cupy.arange(10),
        edge_id_array=eids,
        edge_type_array=etps,
        weight_array=cupy.ones((14,), dtype="float32"),
    )

    sampler = UniformNeighborSampler(
        graph,
        fanout=[-1, -1, -1, -1],
        compression="COO",
        heterogeneous=True,
        vertex_type_offsets=cupy.array([0, 4, 10]),
        num_edge_types=2,
        deduplicate_sources=True,
    )

    out = sampler.sample_from_nodes(
        nodes=cupy.array([4, 5]),
        input_id=cupy.array([5, 10]),
    )

    out = [z for z in out]
    assert len(out) == 1
    out, _, _ = out[0]

    lho = out["label_type_hop_offsets"]

    # Edge type 0
    emap = out["edge_renumber_map"][
        out["edge_renumber_map_offsets"][0] : out["edge_renumber_map_offsets"][1]
    ]

    smap = out["map"][out["renumber_map_offsets"][1] : out["renumber_map_offsets"][2]]

    dmap = out["map"][out["renumber_map_offsets"][0] : out["renumber_map_offsets"][1]]

    # Edge type 0, hop 0
    hop_start = lho[0]
    hop_end = lho[1]

    assert hop_end - hop_start == 2

    e = out["edge_id"][hop_start:hop_end]
    e = emap[e]
    assert sorted(e.tolist()) == [0, 1]

    s = cupy.asarray(smap[out["majors"][hop_start:hop_end]])
    d = cupy.asarray(dmap[out["minors"][hop_start:hop_end]])

    assert sorted(s.tolist()) == [4, 5]
    assert sorted(d.tolist()) == [0, 1]

    # Edge type 0, hop 1
    hop_start = int(lho[1])
    hop_end = int(lho[2])

    assert hop_end - hop_start == 2

    e = out["edge_id"][hop_start:hop_end]
    e = emap[e]
    assert sorted(e.tolist()) == [4, 5]

    s = cupy.asarray(smap[out["majors"][hop_start:hop_end]])
    d = cupy.asarray(dmap[out["minors"][hop_start:hop_end]])

    assert sorted(s.tolist()) == [8, 9]
    assert sorted(d.tolist()) == [0, 3]

    #############################

    # Edge type 1
    emap = out["edge_renumber_map"][
        out["edge_renumber_map_offsets"][1] : out["edge_renumber_map_offsets"][2]
    ]

    smap = out["map"][out["renumber_map_offsets"][1] : out["renumber_map_offsets"][2]]

    dmap = smap

    # Edge type 1, hop 0
    hop_start = lho[2]
    hop_end = lho[3]

    assert hop_end - hop_start == 3

    e = out["edge_id"][hop_start:hop_end]
    e = emap[e]
    assert sorted(e.tolist()) == [5, 6, 7]

    s = cupy.asarray(smap[out["majors"][hop_start:hop_end]])
    d = cupy.asarray(dmap[out["minors"][hop_start:hop_end]])

    assert sorted(s.tolist()) == [4, 5, 5]
    assert sorted(d.tolist()) == [8, 9, 9]

    # Edge type 1, hop 1
    hop_start = lho[3]
    hop_end = lho[4]

    assert hop_end - hop_start == 3

    e = out["edge_id"][hop_start:hop_end]
    e = emap[e]
    assert sorted(e.tolist()) == [0, 1, 2]

    s = cupy.asarray(smap[out["majors"][hop_start:hop_end]])
    d = cupy.asarray(dmap[out["minors"][hop_start:hop_end]])

    assert sorted(s.tolist()) == [8, 8, 9]
    assert sorted(d.tolist()) == [4, 5, 6]
