# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

from cugraph.utilities.utils import import_optional
from cugraph.gnn import cugraph_comms_init

th = import_optional("torch")


def assert_same_node_feats(gs, g):
    assert set(gs.ndata.keys()) == set(g.ndata.keys())
    assert set(gs.ntypes) == set(g.ntypes)

    for key in g.ndata.keys():
        for ntype in g.ntypes:
            if len(g.ntypes) <= 1 or ntype in g.ndata[key]:
                indices = th.arange(0, g.num_nodes(ntype), dtype=g.idtype)

                g_output = g.ndata[key]
                gs_output = gs.ndata[key]

                if len(g.ntypes) > 1:
                    g_output = g_output[ntype]
                    gs_output = gs_output[ntype]

                g_output = g_output[indices]
                gs_output = gs_output[indices]

                equal_t = (gs_output != g_output).sum()
                assert equal_t == 0


def assert_same_num_nodes(gs, g):
    for ntype in g.ntypes:
        assert g.num_nodes(ntype) == gs.num_nodes(ntype)


def assert_same_num_edges_can_etypes(gs, g):
    for can_etype in g.canonical_etypes:
        assert g.num_edges(can_etype) == gs.num_edges(can_etype)


def assert_same_num_edges_etypes(gs, g):
    for etype in g.etypes:
        assert g.num_edges(etype) == gs.num_edges(etype)


def assert_same_edge_feats(gs, g):
    assert set(gs.edata.keys()) == set(g.edata.keys())
    assert set(gs.canonical_etypes) == set(g.canonical_etypes)
    assert set(gs.etypes) == set(g.etypes)

    for key in g.edata.keys():
        for etype in g.canonical_etypes:
            if len(g.etypes) <= 1 or etype in g.edata[key]:
                indices = th.arange(0, g.num_edges(etype), dtype=g.idtype).cuda()
                g_output = g.edata[key]
                gs_output = gs.edata[key]

                if len(g.etypes) > 1:
                    g_output = g_output[etype]
                    gs_output = gs_output[etype]

                g_output = g_output[indices]
                gs_output = gs_output[indices]

                equal_t = (gs_output != g_output).sum().cpu()
                assert equal_t == 0


def init_pytorch_worker(rank, world_size, cugraph_id, init_wholegraph=False):
    import rmm

    rmm.reinitialize(
        devices=rank,
    )

    import cupy

    cupy.cuda.Device(rank).use()
    from rmm.allocators.cupy import rmm_cupy_allocator

    cupy.cuda.set_allocator(rmm_cupy_allocator)

    from cugraph.testing.mg_utils import enable_spilling

    enable_spilling()

    th.cuda.set_device(rank)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    th.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    if init_wholegraph:
        import pylibwholegraph

        pylibwholegraph.torch.initialize.init(
            rank,
            world_size,
            rank,
            world_size,
        )

    cugraph_comms_init(rank=rank, world_size=world_size, uid=cugraph_id, device=rank)
