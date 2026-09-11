# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cupy

from cugraph_pyg.sampler import DistributedNeighborSampler

from pylibcugraph import SGGraph, ResourceHandle, GraphProperties

from cugraph_pyg.utils.imports import import_optional, MissingModule


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

    sampler = DistributedNeighborSampler(
        graph,
        fanout=[-1, -1, -1, -1],
        compression="COO",
        heterogeneous=True,
        vertex_type_offsets=cupy.array([0, 4, 10]),
        num_edge_types=2,
        deduplicate_sources=True,
        biased=False,
    )

    out = sampler.sample_from_nodes(
        nodes=cupy.array([4, 5]),
        input_id=cupy.array([5, 10]),
        metadata={"some_key": "some_value"},
    )

    out = [z for z in out]
    assert len(out) == 1
    out, _, _ = out[0]

    lho = out["label_type_hop_offsets"]
    assert out["some_key"] == "some_value"

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


# ---------------------------------------------------------------------------
# Construction-time validation tests
# ---------------------------------------------------------------------------


def _make_simple_graph():
    """Minimal homogeneous graph for sampler construction tests."""
    props = GraphProperties(is_symmetric=False, is_multigraph=True)
    handle = ResourceHandle()
    srcs = cupy.array([0, 0, 0], dtype="int32")
    dsts = cupy.array([1, 2, 3], dtype="int32")
    etimes = cupy.array([1, 3, 2], dtype="int64")
    return SGGraph(
        handle,
        props,
        srcs,
        dsts,
        edge_id_array=cupy.arange(3, dtype="int32"),
        edge_start_time_array=etimes,
    )


@pytest.mark.sg
def test_dist_sampler_last_invalid_construction():
    """DistributedNeighborSampler raises for unsupported 'last' combinations."""
    graph = _make_simple_graph()

    # 'last' with replacement
    with pytest.raises(ValueError, match="does not support replacement"):
        DistributedNeighborSampler(
            graph,
            fanout=[1],
            temporal=True,
            temporal_strategy="last",
            fixed_window=True,
            with_replacement=True,
        )

    # 'last' with biased sampling
    with pytest.raises(ValueError, match="does not support biased"):
        DistributedNeighborSampler(
            graph,
            fanout=[1],
            temporal=True,
            temporal_strategy="last",
            fixed_window=True,
            biased=True,
        )

    # Unknown strategy name
    with pytest.raises(ValueError, match="Invalid temporal strategy"):
        DistributedNeighborSampler(
            graph,
            fanout=[1],
            temporal=True,
            temporal_strategy="newest",
        )

    # fixed-window with non-monotonically_increasing ordering
    with pytest.raises(ValueError, match="only supports.*monotonically_increasing"):
        DistributedNeighborSampler(
            graph,
            fanout=[1],
            temporal=True,
            fixed_window=True,
            temporal_comparison="strictly_increasing",
        )


@pytest.mark.sg
def test_dist_sampler_fixed_window_auto_temporal():
    """fixed_window=True silently enables temporal even when temporal=False."""
    graph = _make_simple_graph()

    # Should not raise — temporal is force-set to True internally.
    sampler = DistributedNeighborSampler(
        graph,
        fanout=[1],
        temporal=False,
        fixed_window=True,
    )
    # Verify the sampler has the fixed-window flag set by checking that
    # calling sample_batches without seed_start_times raises the right error.
    with pytest.raises(ValueError, match="requires both input_start_time"):
        sampler.sample_batches(
            seeds=cupy.array([0], dtype="int32"),
            seed_times=None,
            batch_id_offsets=cupy.array([0, 1], dtype="int32"),
        )


@pytest.mark.sg
def test_dist_sampler_sample_batches_time_guards():
    """sample_batches enforces mutual-exclusion of time arguments."""
    graph = _make_simple_graph()

    # Non-fixed-window sampler: rejects seed_start_times
    sampler = DistributedNeighborSampler(
        graph,
        fanout=[1],
        temporal=True,
        temporal_comparison="monotonically_increasing",
    )
    with pytest.raises(ValueError, match="require fixed-window"):
        sampler.sample_batches(
            seeds=cupy.array([0], dtype="int32"),
            seed_times=None,
            batch_id_offsets=cupy.array([0, 1], dtype="int32"),
            seed_start_times=cupy.array([0], dtype="int64"),
            seed_end_times=cupy.array([3], dtype="int64"),
        )

    # seed_time combined with seed_start_times is rejected
    with pytest.raises(ValueError, match="cannot be combined"):
        sampler.sample_batches(
            seeds=cupy.array([0], dtype="int32"),
            seed_times=cupy.array([3], dtype="int64"),
            batch_id_offsets=cupy.array([0, 1], dtype="int32"),
            seed_start_times=cupy.array([0], dtype="int64"),
            seed_end_times=cupy.array([3], dtype="int64"),
        )

    # seed_start_times without seed_end_times is rejected
    with pytest.raises(ValueError, match="must be provided together"):
        sampler.sample_batches(
            seeds=cupy.array([0], dtype="int32"),
            seed_times=None,
            batch_id_offsets=cupy.array([0, 1], dtype="int32"),
            seed_start_times=cupy.array([0], dtype="int64"),
        )

    # Fixed-window sampler: rejects plain seed_times
    fw_sampler = DistributedNeighborSampler(
        graph,
        fanout=[1],
        temporal=True,
        fixed_window=True,
    )
    with pytest.raises(ValueError, match="fixed-window sampling requires input_start"):
        fw_sampler.sample_batches(
            seeds=cupy.array([0], dtype="int32"),
            seed_times=cupy.array([3], dtype="int64"),
            batch_id_offsets=cupy.array([0, 1], dtype="int32"),
        )
