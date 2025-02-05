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

import cugraph_dgl
import pylibcugraph
import cupy
import numpy as np

from cugraph.datasets import karate
from cugraph.utilities.utils import import_optional, MissingModule

torch = import_optional("torch")
dgl = import_optional("dgl")


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.skipif(isinstance(dgl, MissingModule), reason="dgl not available")
@pytest.mark.parametrize("direction", ["out", "in"])
def test_graph_make_homogeneous_graph(direction):
    df = karate.get_edgelist()
    df.src = df.src.astype("int64")
    df.dst = df.dst.astype("int64")
    wgt = np.random.random((len(df),))

    graph = cugraph_dgl.Graph()
    num_nodes = max(df.src.max(), df.dst.max()) + 1
    node_x = np.random.random((num_nodes,))

    graph.add_nodes(
        num_nodes, data={"num": torch.arange(num_nodes, dtype=torch.int64), "x": node_x}
    )
    graph.add_edges(df.src, df.dst, {"weight": wgt})
    plc_dgl_graph = graph._graph(direction=direction)

    assert graph.num_nodes() == num_nodes
    assert graph.num_edges() == len(df)
    assert graph.is_homogeneous
    assert not graph.is_multi_gpu

    assert (
        graph.nodes() == torch.arange(num_nodes, dtype=torch.int64, device="cuda")
    ).all()

    emb = graph.nodes[None]["x"]
    assert emb is not None
    assert (emb() == torch.as_tensor(node_x, device="cuda")).all()
    assert (
        graph.nodes[None]["num"]()
        == torch.arange(num_nodes, dtype=torch.int64, device="cuda")
    ).all()

    assert (
        graph.edges("eid", device="cuda")
        == torch.arange(len(df), dtype=torch.int64, device="cuda")
    ).all()
    assert (graph.edges[None]["weight"]() == torch.as_tensor(wgt, device="cuda")).all()

    plc_expected_graph = pylibcugraph.SGGraph(
        pylibcugraph.ResourceHandle(),
        pylibcugraph.GraphProperties(is_multigraph=True, is_symmetric=False),
        df.src if direction == "out" else df.dst,
        df.dst if direction == "out" else df.src,
        vertices_array=cupy.arange(num_nodes, dtype="int64"),
    )

    # Do the expensive check to make sure this test fails if an invalid
    # graph is constructed.
    v_actual, d_in_actual, d_out_actual = pylibcugraph.degrees(
        pylibcugraph.ResourceHandle(),
        plc_dgl_graph,
        source_vertices=cupy.arange(num_nodes, dtype="int64"),
        do_expensive_check=True,
    )

    v_exp, d_in_exp, d_out_exp = pylibcugraph.degrees(
        pylibcugraph.ResourceHandle(),
        plc_expected_graph,
        source_vertices=cupy.arange(num_nodes, dtype="int64"),
        do_expensive_check=True,
    )

    assert (v_actual == v_exp).all()
    assert (d_in_actual == d_in_exp).all()
    assert (d_out_actual == d_out_exp).all()


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.skipif(isinstance(dgl, MissingModule), reason="dgl not available")
@pytest.mark.parametrize("direction", ["out", "in"])
def test_graph_make_heterogeneous_graph(direction, karate_bipartite):
    graph, edges, (num_nodes_group_1, num_nodes_group_2) = karate_bipartite

    assert not graph.is_homogeneous
    assert not graph.is_multi_gpu

    # Verify graph.nodes()
    assert (
        graph.nodes()
        == torch.arange(
            num_nodes_group_1 + num_nodes_group_2, dtype=torch.int64, device="cuda"
        )
    ).all()
    assert (
        graph.nodes("type1")
        == torch.arange(num_nodes_group_1, dtype=torch.int64, device="cuda")
    ).all()
    assert (
        graph.nodes("type2")
        == torch.arange(num_nodes_group_2, dtype=torch.int64, device="cuda")
    ).all()

    # Verify graph.edges()
    assert (
        graph.edges("eid", etype=("type1", "e1", "type1"))
        == torch.arange(
            len(edges["type1", "e1", "type1"]), dtype=torch.int64, device="cuda"
        )
    ).all()
    assert (
        graph.edges("eid", etype=("type1", "e2", "type2"))
        == torch.arange(
            len(edges["type1", "e2", "type2"]), dtype=torch.int64, device="cuda"
        )
    ).all()
    assert (
        graph.edges("eid", etype=("type2", "e3", "type1"))
        == torch.arange(
            len(edges["type2", "e3", "type1"]), dtype=torch.int64, device="cuda"
        )
    ).all()
    assert (
        graph.edges("eid", etype=("type2", "e4", "type2"))
        == torch.arange(
            len(edges["type2", "e4", "type2"]), dtype=torch.int64, device="cuda"
        )
    ).all()

    # Use sampling call to check graph creation
    # This isn't a test of cuGraph sampling with DGL; the options are
    # set to verify the graph only.
    plc_graph = graph._graph(direction)
    sampling_output = pylibcugraph.uniform_neighbor_sample(
        pylibcugraph.ResourceHandle(),
        plc_graph,
        start_list=cupy.arange(num_nodes_group_1 + num_nodes_group_2, dtype="int64"),
        h_fan_out=np.array([1, 1], dtype="int32"),
        with_replacement=False,
        do_expensive_check=True,
        with_edge_properties=True,
        prior_sources_behavior="exclude",
        return_dict=True,
    )

    expected_etypes = {
        0: "e1",
        1: "e2",
        2: "e3",
        3: "e4",
    }
    expected_offsets = {
        0: (0, 0),
        1: (0, num_nodes_group_1),
        2: (num_nodes_group_1, 0),
        3: (num_nodes_group_1, num_nodes_group_1),
    }
    if direction == "in":
        src_col = "minors"
        dst_col = "majors"
    else:
        src_col = "majors"
        dst_col = "minors"

    # Looping over the output verifies that all edges are valid
    # (and therefore, the graph is valid)
    for i, etype in enumerate(sampling_output["edge_type"].tolist()):
        eid = int(sampling_output["edge_id"][i])

        srcs, dsts, eids = graph.edges(
            "all", etype=expected_etypes[etype], device="cpu"
        )

        assert eids[eid] == eid
        assert (
            srcs[eid] == int(sampling_output[src_col][i]) - expected_offsets[etype][0]
        )
        assert (
            dsts[eid] == int(sampling_output[dst_col][i]) - expected_offsets[etype][1]
        )


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.skipif(isinstance(dgl, MissingModule), reason="dgl not available")
@pytest.mark.parametrize("direction", ["out", "in"])
def test_graph_find(direction, karate_bipartite):
    graph, edges, _ = karate_bipartite

    # force direction generation to make sure in case is tested
    graph._graph(direction)

    assert not graph.is_homogeneous
    assert not graph.is_multi_gpu

    srcs, dsts = graph.find_edges(
        torch.as_tensor(
            [0, len(edges["type1", "e1", "type1"]) - 1, 999], dtype=torch.int64
        ),
        ("type1", "e1", "type1"),
    )
    assert (
        srcs[[0, 1]] == torch.tensor([1, 6], device="cuda", dtype=torch.int64)
    ).all()
    assert (
        dsts[[0, 1]] == torch.tensor([0, 16], device="cuda", dtype=torch.int64)
    ).all()
    assert srcs[2] < 0 and dsts[2] < 0

    srcs, dsts = graph.find_edges(
        torch.as_tensor(
            [0, len(edges["type1", "e2", "type2"]) - 1, 999], dtype=torch.int64
        ),
        ("type1", "e2", "type2"),
    )
    assert (
        srcs[[0, 1]] == torch.tensor([0, 15], device="cuda", dtype=torch.int64)
    ).all()
    assert (
        dsts[[0, 1]] == torch.tensor([0, 16], device="cuda", dtype=torch.int64)
    ).all()
    assert srcs[2] < 0 and dsts[2] < 0

    srcs, dsts = graph.find_edges(
        torch.as_tensor(
            [0, len(edges["type2", "e3", "type1"]) - 1, 999], dtype=torch.int64
        ),
        ("type2", "e3", "type1"),
    )
    assert (
        srcs[[0, 1]] == torch.tensor([0, 16], device="cuda", dtype=torch.int64)
    ).all()
    assert (
        dsts[[0, 1]] == torch.tensor([0, 15], device="cuda", dtype=torch.int64)
    ).all()
    assert srcs[2] < 0 and dsts[2] < 0

    srcs, dsts = graph.find_edges(
        torch.as_tensor(
            [0, len(edges["type2", "e4", "type2"]) - 1, 999], dtype=torch.int64
        ),
        ("type2", "e4", "type2"),
    )
    assert (
        srcs[[0, 1]] == torch.tensor([15, 15], device="cuda", dtype=torch.int64)
    ).all()
    assert (
        dsts[[0, 1]] == torch.tensor([1, 16], device="cuda", dtype=torch.int64)
    ).all()
    assert srcs[2] < 0 and dsts[2] < 0


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.skipif(isinstance(dgl, MissingModule), reason="dgl not available")
@pytest.mark.parametrize("exclude_self_loops", [True, False])
@pytest.mark.parametrize("num_samples", [2, 11])
def test_graph_uniform_negative_sample(
    karate_bipartite, exclude_self_loops, num_samples
):
    graph, edges, _ = karate_bipartite

    for etype in graph.canonical_etypes:
        src_neg, dst_neg = graph.global_uniform_negative_sampling(
            num_samples,
            exclude_self_loops=exclude_self_loops,
            etype=etype,
        )

        assert len(src_neg) == len(dst_neg)
        assert len(src_neg) <= num_samples

        assert (src_neg >= 0).all()
        assert (dst_neg >= 0).all()

        assert (src_neg < graph.num_nodes(etype[0])).all()
        assert (dst_neg < graph.num_nodes(etype[2])).all()

        if exclude_self_loops:
            assert (src_neg == dst_neg).sum() == 0

        for i in range(len(src_neg)):
            s = int(src_neg[i])
            d = int(dst_neg[i])
            assert ((edges[etype].src == s) & (edges[etype].dst == d)).sum() == 0
