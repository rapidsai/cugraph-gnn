# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cugraph.datasets import karate
from cugraph_pyg.utils.imports import import_optional, MissingModule

from cugraph_pyg.tensor import DistMatrix
from cugraph_pyg.data import FeatureStore, GraphStore
from cugraph_pyg.loader import NeighborLoader

torch = import_optional("torch")
torch_geometric = import_optional("torch_geometric")


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
@pytest.mark.parametrize("location", ["cpu", "cuda"])
def test_graph_store_basic_api(single_pytorch_worker, location):
    df = karate.get_edgelist()
    src = torch.as_tensor(df["src"], device="cuda")
    dst = torch.as_tensor(df["dst"], device="cuda")

    ei = torch.stack([dst, src])

    num_nodes = karate.number_of_nodes()

    graph_store = GraphStore(location=location)
    graph_store.put_edge_index(
        ei, ("person", "knows", "person"), "coo", False, (num_nodes, num_nodes)
    )

    rei = graph_store.get_edge_index(("person", "knows", "person"), "coo")

    assert (ei == rei).all()

    edge_attrs = graph_store.get_all_edge_attrs()
    assert len(edge_attrs) == 1

    graph_store.remove_edge_index(("person", "knows", "person"), "coo")
    edge_attrs = graph_store.get_all_edge_attrs()
    assert len(edge_attrs) == 0

    dist_matrix = DistMatrix(src=(dst, src), device=location, format="coo")
    graph_store.put_edge_index(
        dist_matrix,
        ("person", "follows", "person"),
        "coo",
        False,
        (num_nodes, num_nodes),
    )

    stored_matrix = graph_store._GraphStore__edge_indices[
        ("person", "follows", "person")
    ]
    assert stored_matrix is dist_matrix

    rei = graph_store.get_edge_index(("person", "follows", "person"), "coo")
    assert (ei == rei).all()

    graph_store.put_edge_index(
        (dist_matrix._row, dist_matrix._col),
        ("person", "likes", "person"),
        "coo",
        False,
        (num_nodes, num_nodes),
    )

    stored_matrix = graph_store._GraphStore__edge_indices[("person", "likes", "person")]
    assert stored_matrix._row is dist_matrix._row
    assert stored_matrix._col is dist_matrix._col


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_graph_store_finalize(single_pytorch_worker):
    df = karate.get_edgelist()
    src = torch.as_tensor(df["src"], device="cuda")
    dst = torch.as_tensor(df["dst"], device="cuda")
    edge_index = torch.stack([dst, src])
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
    loader = NeighborLoader(
        (feature_store, graph_store),
        [5, 5],
        input_nodes=torch.arange(num_nodes),
        batch_size=num_nodes,
    )

    batch = next(iter(loader))
    assert isinstance(batch, torch_geometric.data.Data)
    assert (feature_store["person", "feat", None][batch.n_id] == batch.feat).all()
