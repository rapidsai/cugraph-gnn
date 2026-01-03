# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cugraph.datasets import karate
from cugraph_pyg.utils.imports import import_optional, MissingModule

import cugraph_pyg
from cugraph_pyg.data import GraphStore, FeatureStore
from cugraph_pyg.loader import NeighborLoader

torch = import_optional("torch")
torch_geometric = import_optional("torch_geometric")


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_neighbor_loader(single_pytorch_worker):
    """
    Basic e2e test that covers loading and sampling.
    """

    df = karate.get_edgelist()
    src = torch.as_tensor(df["src"], device="cuda")
    dst = torch.as_tensor(df["dst"], device="cuda")

    ei = torch.stack([dst, src])

    num_nodes = karate.number_of_nodes()

    graph_store = GraphStore()
    graph_store.put_edge_index(
        ei, ("person", "knows", "person"), "coo", False, (num_nodes, num_nodes)
    )

    feature_store = FeatureStore()
    feature_store["person", "feat", None] = torch.randint(128, (34, 16))

    loader = NeighborLoader(
        (feature_store, graph_store),
        [5, 5],
        input_nodes=torch.arange(34),
    )

    for batch in loader:
        assert isinstance(batch, torch_geometric.data.Data)
        assert (feature_store["person", "feat", None][batch.n_id] == batch.feat).all()


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_neighbor_loader_biased(single_pytorch_worker):
    eix = torch.tensor(
        [
            [3, 4, 5],
            [0, 1, 2],
        ]
    )

    num_nodes = 6

    graph_store = GraphStore()
    graph_store.put_edge_index(
        eix, ("person", "knows", "person"), "coo", False, (num_nodes, num_nodes)
    )

    feature_store = FeatureStore()
    feature_store["person", "feat", None] = torch.randint(128, (6, 12))
    feature_store[("person", "knows", "person"), "bias", None] = torch.tensor(
        [0, 12, 14], dtype=torch.float32
    )

    loader = NeighborLoader(
        (feature_store, graph_store),
        [1],
        input_nodes=torch.tensor([0, 1, 2], dtype=torch.int64),
        batch_size=3,
        weight_attr="bias",
    )

    out = list(iter(loader))
    assert len(out) == 1
    out = out[0]

    assert out.edge_index.shape[1] == 2
    assert (out.edge_index.cpu() == torch.tensor([[3, 4], [1, 2]])).all()


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
@pytest.mark.parametrize("num_nodes", [10, 25])
@pytest.mark.parametrize("num_edges", [64, 128])
@pytest.mark.parametrize("batch_size", [2, 4])
@pytest.mark.parametrize("select_edges", [16, 32])
@pytest.mark.parametrize("depth", [1, 3])
@pytest.mark.parametrize("num_neighbors", [1, 4])
def test_link_neighbor_loader_basic(
    num_nodes,
    num_edges,
    batch_size,
    select_edges,
    num_neighbors,
    depth,
    single_pytorch_worker,
):
    graph_store = GraphStore()
    feature_store = torch_geometric.data.HeteroData()

    eix = torch.randperm(num_edges)[:select_edges]
    graph_store[("n", "e", "n"), "coo", False, (num_nodes, num_nodes)] = torch.stack(
        [
            torch.randint(0, num_nodes, (num_edges,)),
            torch.randint(0, num_nodes, (num_edges,)),
        ]
    )

    elx = graph_store[("n", "e", "n"), "coo"][:, eix]
    loader = cugraph_pyg.loader.LinkNeighborLoader(
        (feature_store, graph_store),
        num_neighbors=[num_neighbors] * depth,
        edge_label_index=elx,
        batch_size=batch_size,
        shuffle=False,
    )

    elx = torch.tensor_split(elx, eix.numel() // batch_size, dim=1)
    for i, batch in enumerate(loader):
        assert (
            batch.input_id.cpu() == torch.arange(i * batch_size, (i + 1) * batch_size)
        ).all()
        assert (elx[i].cpu() == batch.n_id[batch.edge_label_index.cpu()].cpu()).all()


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
@pytest.mark.parametrize("batch_size", [1, 2])
def test_link_neighbor_loader_negative_sampling_basic(
    batch_size, single_pytorch_worker
):
    num_edges = 62
    num_nodes = 19
    select_edges = 17

    graph_store = GraphStore()
    feature_store = torch_geometric.data.HeteroData()

    eix = torch.randperm(num_edges)[:select_edges]
    graph_store[("n", "e", "n"), "coo", False, (num_nodes, num_nodes)] = torch.stack(
        [
            torch.randint(0, num_nodes, (num_edges,)),
            torch.randint(0, num_nodes, (num_edges,)),
        ]
    )

    elx = graph_store[("n", "e", "n"), "coo"][:, eix]
    loader = cugraph_pyg.loader.LinkNeighborLoader(
        (feature_store, graph_store),
        num_neighbors=[3, 3, 3],
        edge_label_index=elx,
        batch_size=batch_size,
        neg_sampling="binary",
        shuffle=False,
    )

    elx = torch.tensor_split(elx, eix.numel() // batch_size, dim=1)
    for i, batch in enumerate(loader):
        assert batch.edge_label[0] == 1.0


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
@pytest.mark.parametrize("batch_size", [1, 2])
def test_link_neighbor_loader_negative_sampling_uneven(
    batch_size, single_pytorch_worker
):
    num_edges = 62
    num_nodes = 19
    select_edges = 17

    graph_store = GraphStore()
    feature_store = torch_geometric.data.HeteroData()

    eix = torch.randperm(num_edges)[:select_edges]
    graph_store[("n", "e", "n"), "coo", False, (num_nodes, num_nodes)] = torch.stack(
        [
            torch.randint(0, num_nodes, (num_edges,)),
            torch.randint(0, num_nodes, (num_edges,)),
        ]
    )

    elx = graph_store[("n", "e", "n"), "coo"][:, eix]
    loader = cugraph_pyg.loader.LinkNeighborLoader(
        (feature_store, graph_store),
        num_neighbors=[3, 3, 3],
        edge_label_index=elx,
        batch_size=batch_size,
        neg_sampling=torch_geometric.sampler.NegativeSampling("binary", amount=0.1),
        shuffle=False,
    )

    elx = torch.tensor_split(elx, eix.numel() // batch_size, dim=1)
    for i, batch in enumerate(loader):
        assert batch.edge_label[0] == 1.0


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_neighbor_loader_hetero_basic(single_pytorch_worker):
    src = torch.tensor([0, 1, 2, 4, 3, 4, 5, 5])  # paper
    dst = torch.tensor([4, 5, 4, 3, 2, 1, 0, 1])  # paper

    asrc = torch.tensor([0, 1, 2, 3, 3, 0])  # author
    adst = torch.tensor([0, 1, 2, 3, 4, 5])  # paper

    num_authors = 4
    num_papers = 6

    graph_store = GraphStore()
    feature_store = FeatureStore()

    graph_store[("paper", "cites", "paper"), "coo", False, (num_papers, num_papers)] = [
        src,
        dst,
    ]
    graph_store[
        ("author", "writes", "paper"), "coo", False, (num_authors, num_papers)
    ] = [asrc, adst]

    from cugraph_pyg.loader import NeighborLoader

    loader = NeighborLoader(
        (feature_store, graph_store),
        num_neighbors={
            ("paper", "cites", "paper"): [1, 1],
            ("author", "writes", "paper"): [1, 1],
        },
        input_nodes=("paper", torch.tensor([0, 1])),
        batch_size=2,
    )

    out = next(iter(loader))

    ei_out = out["paper"].n_id[out["paper", "cites", "paper"].edge_index.cpu()]
    assert (
        src[out["paper", "cites", "paper"].e_id.cpu()].cpu() == ei_out[0].cpu()
    ).all()
    assert (
        dst[out["paper", "cites", "paper"].e_id.cpu()].cpu() == ei_out[1].cpu()
    ).all()

    ej_out = torch.stack(
        [
            out["author"].n_id[out["author", "writes", "paper"].edge_index[0].cpu()],
            out["paper"].n_id[out["author", "writes", "paper"].edge_index[1].cpu()],
        ]
    )
    assert (
        asrc[out["author", "writes", "paper"].e_id.cpu()].cpu() == ej_out[0].cpu()
    ).all()
    assert (
        adst[out["author", "writes", "paper"].e_id.cpu()].cpu() == ej_out[1].cpu()
    ).all()


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_neighbor_loader_hetero_single_etype(single_pytorch_worker):
    src = torch.tensor([0, 1, 2, 4, 3, 4, 5, 5])  # paper
    dst = torch.tensor([4, 5, 4, 3, 2, 1, 0, 1])  # paper

    asrc = torch.tensor([0, 1, 2, 3, 3, 0])  # author
    adst = torch.tensor([0, 1, 2, 3, 4, 5])  # paper

    num_authors = 4
    num_papers = 6

    graph_store = GraphStore()
    feature_store = torch_geometric.data.HeteroData()

    graph_store[("paper", "cites", "paper"), "coo", False, (num_papers, num_papers)] = [
        src,
        dst,
    ]
    graph_store[
        ("author", "writes", "paper"), "coo", False, (num_authors, num_papers)
    ] = [asrc, adst]

    from cugraph_pyg.loader import NeighborLoader

    loader = NeighborLoader(
        (feature_store, graph_store),
        num_neighbors={
            ("paper", "cites", "paper"): [1, 1],
        },
        input_nodes=("paper", torch.tensor([0, 1])),
        batch_size=2,
    )

    out = next(iter(loader))

    assert out["author"].n_id.numel() == 0
    assert out["author", "writes", "paper"].edge_index.numel() == 0
    assert out["author", "writes", "paper"].num_sampled_edges.tolist() == [0, 0]


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_neighbor_loader_hetero_linkpred(single_pytorch_worker):
    src = torch.tensor([0, 1, 2, 4, 3, 4, 5, 5])  # paper
    dst = torch.tensor([4, 5, 4, 3, 2, 1, 0, 1])  # paper

    asrc = torch.tensor([0, 1, 2, 3, 3, 0])  # author
    adst = torch.tensor([0, 1, 2, 3, 4, 5])  # paper

    num_authors = 4
    num_papers = 6

    graph_store = GraphStore()
    feature_store = torch_geometric.data.HeteroData()

    graph_store[("paper", "cites", "paper"), "coo", False, (num_papers, num_papers)] = [
        src,
        dst,
    ]
    graph_store[
        ("author", "writes", "paper"), "coo", False, (num_authors, num_papers)
    ] = [asrc, adst]

    from cugraph_pyg.loader import LinkNeighborLoader

    loader = LinkNeighborLoader(
        (feature_store, graph_store),
        num_neighbors={
            ("paper", "cites", "paper"): [2, 2],
            ("author", "writes", "paper"): [2, 2],
        },
        edge_label_index=(("author", "writes", "paper"), torch.stack([asrc, adst])),
        batch_size=5,
    )

    out = next(iter(loader))

    assert out["paper"].n_id.numel() == 6
    assert out["paper"].n_id.tolist() == [0, 1, 2, 3, 4, 5]
    # FIXME test for the num_nodes attribute
    # assert out['author'].num_nodes == 4

    assert out["author"].n_id.numel() == 4
    assert out["author"].n_id.tolist() == [0, 1, 2, 3]
    # FIXME test for the num_nodes attribute
    # assert out["paper"].num_nodes == 4

    assert out["paper"].num_sampled_nodes.tolist() == [5, 1, 0]
    assert out["author"].num_sampled_nodes.tolist() == [4, 0, 0]

    assert out["paper", "cites", "paper"].edge_index.shape == torch.Size([2, 8])
    assert out["paper", "cites", "paper"].num_sampled_edges.tolist() == [7, 1]
    assert "edge_label_index" not in out["paper", "cites", "paper"]

    assert out["author", "writes", "paper"].edge_index.shape == torch.Size([2, 6])
    assert out["author", "writes", "paper"].num_sampled_edges.tolist() == [5, 1]

    assert list(out["author", "writes", "paper"].edge_label_index.shape) == [2, 5]
    assert out["author", "writes", "paper"].edge_label_index.tolist()[0] == [
        0,
        1,
        2,
        3,
        3,
    ]
    assert out["author", "writes", "paper"].edge_label_index.tolist()[1] == [
        0,
        1,
        2,
        3,
        4,
    ]


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_neighbor_loader_hetero_linkpred_bidirectional(single_pytorch_worker):
    num_users = 9
    num_merchants = 5

    src = torch.tensor([1, 5, 5, 8, 1, 1], dtype=torch.long)
    dst = torch.tensor([4, 2, 3, 1, 0, 4], dtype=torch.long)

    feature_store = FeatureStore()
    graph_store = GraphStore()

    graph_store[
        ("user", "to", "merchant"), "coo", False, (num_users, num_merchants)
    ] = torch.stack([src, dst], dim=0)
    graph_store[
        ("merchant", "rev_to", "user"), "coo", False, (num_merchants, num_users)
    ] = torch.stack([dst, src], dim=0)

    # use nonexistent edges for more robustness
    from cugraph_pyg.loader import LinkNeighborLoader

    eli = torch.tensor([[0, 5, 8, 1, 7, 2], [4, 4, 2, 3, 1, 0]])
    loader = LinkNeighborLoader(
        data=(feature_store, graph_store),
        num_neighbors={
            ("user", "to", "merchant"): [2, 2],
            ("merchant", "rev_to", "user"): [2, 2],
        },
        edge_label_index=(
            ("user", "to", "merchant"),
            eli,
        ),
        edge_label=None,
        batch_size=2,
        shuffle=False,
    )

    for i, batch in enumerate(loader):
        eli_i = eli[:, i * 2 : (i + 1) * 2]

        r_i = torch.stack(
            [
                batch["user"]
                .n_id[batch["user", "to", "merchant"].edge_label_index[0].cpu()]
                .cpu(),
                batch["merchant"]
                .n_id[batch["user", "to", "merchant"].edge_label_index[1].cpu()]
                .cpu(),
            ]
        )

        assert (r_i == eli_i).all()

    assert i == 2


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_neighbor_loader_hetero_linkpred_bidirectional_v2(single_pytorch_worker):
    num_nodes_n1 = 15
    num_nodes_n2 = 8

    ei = torch.tensor(
        [
            [14, 14, 0, 7, 8, 7, 13, 13, 3, 13, 14, 6, 3, 14, 3, 1, 11, 11, 13, 4],
            [7, 0, 3, 1, 0, 0, 0, 4, 2, 3, 3, 1, 4, 3, 0, 6, 5, 1, 4, 4],
        ]
    )

    feature_store = FeatureStore()
    graph_store = GraphStore()

    graph_store[("n1", "e", "n2"), "coo", False, (num_nodes_n1, num_nodes_n2)] = ei
    graph_store[("n2", "f", "n1"), "coo", False, (num_nodes_n2, num_nodes_n1)] = (
        ei.flip(0)
    )

    from cugraph_pyg.loader import LinkNeighborLoader

    eli = torch.tensor(
        [
            [3, 14, 4, 0, 14, 13, 8, 13, 6, 11, 14, 13, 13, 1, 11, 7],
            [2, 0, 4, 3, 3, 4, 0, 0, 1, 1, 3, 4, 3, 6, 5, 0],
        ]
    )
    loader = LinkNeighborLoader(
        data=(feature_store, graph_store),
        num_neighbors={
            ("n1", "e", "n2"): [2, 2],
            ("n2", "f", "n1"): [2, 2],
        },
        edge_label_index=(
            ("n1", "e", "n2"),
            eli,
        ),
        edge_label=None,
        batch_size=2,
        shuffle=False,
    )

    for i, batch in enumerate(loader):
        eli_i = eli[:, i * 2 : (i + 1) * 2]

        r_i = torch.stack(
            [
                batch["n1"]
                .n_id[batch["n1", "e", "n2"].edge_label_index[0].cpu()]
                .cpu(),
                batch["n2"]
                .n_id[batch["n1", "e", "n2"].edge_label_index[1].cpu()]
                .cpu(),
            ]
        )

        assert (r_i == eli_i).all()

    assert i == 7


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
def test_neighbor_loader_hetero_linkpred_bidirectional_three_types(
    single_pytorch_worker,
):
    num_nodes_n1 = 15
    num_nodes_n2 = 8
    num_nodes_n3 = 10

    ei = torch.tensor(
        [
            [14, 14, 0, 7, 8, 7, 13, 13, 3, 13, 14, 6, 3, 14, 3, 1, 11, 11, 13, 4],
            [7, 0, 3, 1, 0, 0, 0, 4, 2, 3, 3, 1, 4, 3, 0, 6, 5, 1, 4, 4],
        ]
    )
    ei_13 = torch.tensor(
        [
            [1, 3, 5, 6, 8, 14, 14],
            [2, 4, 6, 8, 9, 0, 1],
        ]
    )
    ei_23 = torch.tensor(
        [
            [7, 0, 3, 2, 2, 1, 1, 5, 4, 2],
            [9, 8, 1, 2, 3, 9, 8, 4, 6, 5],
        ]
    )

    feature_store = FeatureStore()
    graph_store = GraphStore()

    graph_store[("n1", "e", "n2"), "coo", False, (num_nodes_n1, num_nodes_n2)] = ei
    graph_store[("n2", "f", "n1"), "coo", False, (num_nodes_n2, num_nodes_n1)] = (
        ei.flip(0)
    )
    graph_store[("n1", "g", "n3"), "coo", False, (num_nodes_n1, num_nodes_n3)] = ei_13
    graph_store[("n2", "h", "n3"), "coo", False, (num_nodes_n2, num_nodes_n3)] = ei_23
    graph_store[("n3", "i", "n1"), "coo", False, (num_nodes_n3, num_nodes_n1)] = (
        ei_13.flip(0)
    )
    graph_store[("n3", "j", "n2"), "coo", False, (num_nodes_n3, num_nodes_n2)] = (
        ei_23.flip(0)
    )

    from cugraph_pyg.loader import LinkNeighborLoader

    eli = torch.tensor(
        [
            [3, 14, 4, 0, 14, 13, 8, 13, 6, 11, 14, 13, 13, 1, 11, 7],
            [2, 0, 4, 3, 3, 4, 0, 0, 1, 1, 3, 4, 3, 6, 5, 0],
        ]
    )
    loader = LinkNeighborLoader(
        data=(feature_store, graph_store),
        num_neighbors={
            ("n1", "e", "n2"): [2, 2],
            ("n2", "f", "n1"): [2, 2],
            ("n1", "g", "n3"): [2, 2],
            ("n2", "h", "n3"): [2, 2],
            ("n3", "i", "n1"): [2, 2],
            ("n3", "j", "n2"): [2, 2],
        },
        edge_label_index=(
            ("n1", "e", "n2"),
            eli,
        ),
        edge_label=None,
        batch_size=2,
        shuffle=False,
    )

    for i, batch in enumerate(loader):
        eli_i = eli[:, i * 2 : (i + 1) * 2]

        r_i = torch.stack(
            [
                batch["n1"]
                .n_id[batch["n1", "e", "n2"].edge_label_index[0].cpu()]
                .cpu(),
                batch["n2"]
                .n_id[batch["n1", "e", "n2"].edge_label_index[1].cpu()]
                .cpu(),
            ]
        )

        assert (r_i == eli_i).all()

    assert i == 7


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.sg
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("neg_sampling_mode", ["binary", "triplet"])
@pytest.mark.parametrize("amount", [1, 2])
def test_link_neighbor_loader_hetero_negative_sampling(
    batch_size, neg_sampling_mode, amount, single_pytorch_worker
):
    """
    Test negative sampling for heterogeneous graphs with different edge types.
    """
    # Create a heterogeneous graph with paper-author relationships
    src_paper = torch.tensor([0, 1, 2, 4, 3, 4, 5, 5])  # paper
    dst_paper = torch.tensor([4, 5, 4, 3, 2, 1, 0, 1])  # paper

    asrc = torch.tensor([0, 1, 2, 3, 3, 0])  # author
    adst = torch.tensor([0, 1, 2, 3, 4, 5])  # paper

    num_authors = 4
    num_papers = 6

    graph_store = GraphStore()
    feature_store = FeatureStore()

    # Add paper-paper citations
    graph_store[("paper", "cites", "paper"), "coo", False, (num_papers, num_papers)] = [
        src_paper,
        dst_paper,
    ]
    # Add author-paper relationships
    graph_store[
        ("author", "writes", "paper"), "coo", False, (num_authors, num_papers)
    ] = [asrc, adst]

    # Create edge label index for author-paper relationships
    edge_label_index = torch.stack([asrc, adst])

    # Test both binary and triplet negative sampling
    if neg_sampling_mode == "binary":
        neg_sampling = torch_geometric.sampler.NegativeSampling(
            "binary", amount=float(amount)
        )
    else:
        neg_sampling = torch_geometric.sampler.NegativeSampling(
            "triplet", amount=float(amount)
        )

    loader = cugraph_pyg.loader.LinkNeighborLoader(
        (feature_store, graph_store),
        num_neighbors={
            ("paper", "cites", "paper"): [2, 2],
            ("author", "writes", "paper"): [2, 2],
        },
        edge_label_index=(("author", "writes", "paper"), edge_label_index),
        batch_size=batch_size,
        neg_sampling=neg_sampling,
        shuffle=False,
    )

    # Test that the loader produces batches with proper negative sampling
    for i, batch in enumerate(loader):
        # Check that we have the expected edge label index structure
        assert [("author", "writes", "paper")] == list(
            batch.edge_label_index_dict.keys()
        )
        assert [("author", "writes", "paper")] == list(batch.edge_label_dict.keys())

        # Should have both positive (1.0) and negative (0.0) labels
        edge_labels = batch["author", "writes", "paper"].edge_label
        assert torch.any(edge_labels == 1.0)
        assert torch.any(edge_labels == 0.0)
        assert (edge_labels == 0.0).sum() == amount * (edge_labels == 1.0).sum()

        # Verify that the edge label index has the correct shape
        edge_label_idx = batch["author", "writes", "paper"].edge_label_index
        assert edge_label_idx.shape[0] == 2  # Should be [2, num_edges]
        assert edge_label_idx.shape[1] > 0  # Should have some edges

        # Verify that the edge labels correspond to the edge label index
        assert edge_labels.shape[0] == edge_label_idx.shape[1]

        # Check that node IDs are valid
        assert batch["author"].n_id.numel() > 0
        assert batch["paper"].n_id.numel() > 0

        # Verify that edge label index uses valid node IDs
        author_n_ids = batch["author"].n_id
        paper_n_ids = batch["paper"].n_id

        # All source nodes in edge_label_index should be in author.n_id
        src_nodes = edge_label_idx[0]
        assert torch.all(torch.isin(src_nodes.cpu(), torch.arange(len(author_n_ids))))

        # All destination nodes in edge_label_index should be in paper.n_id
        dst_nodes = edge_label_idx[1]
        assert torch.all(torch.isin(dst_nodes.cpu(), torch.arange(len(paper_n_ids))))

    # Verify we processed all batches
    assert i >= 0  # At least one batch should be processed


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.parametrize("biased", [True, False])
@pytest.mark.sg
def test_neighbor_loader_temporal_simple(single_pytorch_worker, biased):
    """
    Test negative sampling for heterogeneous graphs with different edge types.
    """
    # Create a homogeneous graph with paper-paper citations
    src_cite = torch.tensor([3, 2, 1, 2])  # paper
    dst_cite = torch.tensor([2, 1, 0, 0])  # paper
    tme_cite = torch.tensor([0, 1, 2, 0])  # time

    num_papers = 4

    graph_store = GraphStore()
    feature_store = FeatureStore()

    # Add paper-paper citations
    graph_store[("paper", "cites", "paper"), "coo", False, (num_papers, num_papers)] = [
        dst_cite,
        src_cite,
    ]

    feature_store[("paper", "cites", "paper"), "time", None] = tme_cite
    feature_store[("paper", "cites", "paper"), "bias", None] = torch.tensor(
        [1.0] * src_cite.numel(), device="cuda"
    )

    loader = cugraph_pyg.loader.NeighborLoader(
        (feature_store, graph_store),
        num_neighbors=[2, 2, 2],
        batch_size=1,
        input_nodes=torch.tensor([3]),
        input_time=torch.tensor([-1]),
        time_attr="time",
        shuffle=False,
        weight_attr="bias" if biased else None,
        temporal_comparison="strictly_increasing",
    )

    out = next(iter(loader))
    assert out.n_id.tolist() == [3, 2, 1, 0]
    assert out.e_id.tolist() == [0, 1, 2]
    assert out.num_sampled_nodes.tolist() == [1, 1, 1, 1]
    assert out.num_sampled_edges.tolist() == [1, 1, 1]


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.parametrize("biased", [True, False])
@pytest.mark.sg
def test_neighbor_loader_temporal_hetero(single_pytorch_worker, biased):
    """
    Test negative sampling for heterogeneous graphs with different edge types.
    """
    # Create a homogeneous graph with paper-paper citations
    src_cite = torch.tensor([3, 2, 1, 2])  # paper
    dst_cite = torch.tensor([2, 1, 0, 0])  # paper
    tme_cite = torch.tensor([0, 1, 2, 0])  # time

    src_author = torch.tensor([3, 2, 2, 1, 3, 2, 0])  # paper
    dst_author = torch.tensor([0, 0, 1, 1, 2, 2, 2])  # author
    tme_author = torch.tensor([0, 0, 1, 0, 2, 1, 1])  # time

    num_papers = 4
    num_authors = 3
    graph_store = GraphStore()
    feature_store = FeatureStore()

    # Add paper-paper citations
    graph_store[("paper", "cites", "paper"), "coo", False, (num_papers, num_papers)] = [
        dst_cite,
        src_cite,
    ]

    graph_store[
        ("author", "writes", "paper"), "coo", False, (num_authors, num_papers)
    ] = [
        dst_author,
        src_author,
    ]

    feature_store[("paper", "cites", "paper"), "time", None] = tme_cite
    feature_store[("author", "writes", "paper"), "time", None] = tme_author

    feature_store[("paper", "cites", "paper"), "bias", None] = torch.tensor(
        [1.0] * src_cite.numel(), device="cuda"
    )
    feature_store[("author", "writes", "paper"), "bias", None] = torch.tensor(
        [1.0] * src_author.numel(), device="cuda"
    )

    loader = cugraph_pyg.loader.NeighborLoader(
        (feature_store, graph_store),
        num_neighbors={
            ("paper", "cites", "paper"): [2, 2, 2],
            ("author", "writes", "paper"): [2, 2, 0],
        },
        batch_size=1,
        input_nodes=("paper", torch.tensor([3])),
        input_time=torch.tensor([-1]),
        time_attr="time",
        weight_attr="bias" if biased else None,
        shuffle=False,
        temporal_comparison="strictly_increasing",
    )

    out = next(iter(loader))

    assert sorted(out["author"].n_id.tolist()) == [0, 1, 2]
    assert out["paper"].n_id.tolist() == [3, 2, 1, 0]

    assert sorted(out["author", "writes", "paper"].e_id.tolist()) == [0, 2, 4, 5]
    assert out["author", "writes", "paper"].num_sampled_edges.tolist() == [2, 2, 0]


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.parametrize("biased", [True, False])
@pytest.mark.sg
def test_neighbor_loader_temporal_linkpred_homogeneous(single_pytorch_worker, biased):
    """
    Test negative sampling for heterogeneous graphs with different edge types.
    """
    # Create a homogeneous graph with paper-paper citations
    src_cite = torch.tensor([3, 2, 1, 2])  # paper
    dst_cite = torch.tensor([2, 1, 0, 0])  # paper
    tme_cite = torch.tensor([0, 1, 2, 0])  # time

    num_papers = 4
    graph_store = GraphStore()
    feature_store = FeatureStore()

    # Add paper-paper citations
    graph_store[("paper", "cites", "paper"), "coo", False, (num_papers, num_papers)] = [
        dst_cite,
        src_cite,
    ]

    feature_store[("paper", "cites", "paper"), "time", None] = tme_cite

    feature_store[("paper", "cites", "paper"), "bias", None] = torch.tensor(
        [1.0] * src_cite.numel(), device="cuda"
    )

    loader = cugraph_pyg.loader.LinkNeighborLoader(
        (feature_store, graph_store),
        num_neighbors=[2, 2, 2],
        batch_size=1,
        edge_label_index=torch.tensor([[3], [3]]),
        edge_label_time=torch.tensor([-1]),
        time_attr="time",
        weight_attr="bias" if biased else None,
        shuffle=False,
        temporal_comparison="strictly_increasing",
    )

    out = next(iter(loader))

    assert out.n_id.tolist() == [3, 2, 1, 0]

    # FIXME resolve issues with num_sampled_nodes


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.parametrize("biased", [True, False])
@pytest.mark.sg
def test_neighbor_loader_temporal_linkpred_heterogeneous(single_pytorch_worker, biased):
    """
    Test negative sampling for heterogeneous graphs with different edge types.
    """
    # Create a homogeneous graph with paper-paper citations
    src_cite = torch.tensor([3, 2, 1, 2])  # paper
    dst_cite = torch.tensor([2, 1, 0, 0])  # paper
    tme_cite = torch.tensor([0, 1, 2, 0])  # time

    src_author = torch.tensor([3, 2, 2, 1, 3, 2, 0])  # paper
    dst_author = torch.tensor([0, 0, 1, 1, 2, 2, 2])  # author
    tme_author = torch.tensor([0, 0, 1, 0, 2, 1, 1])  # time

    num_papers = 4
    num_authors = 3
    graph_store = GraphStore()
    feature_store = FeatureStore()

    # Add paper-paper citations
    graph_store[("paper", "cites", "paper"), "coo", False, (num_papers, num_papers)] = [
        dst_cite,
        src_cite,
    ]

    graph_store[
        ("author", "writes", "paper"), "coo", False, (num_authors, num_papers)
    ] = [
        dst_author,
        src_author,
    ]

    feature_store[("paper", "cites", "paper"), "time", None] = tme_cite
    feature_store[("author", "writes", "paper"), "time", None] = tme_author

    feature_store[("paper", "cites", "paper"), "bias", None] = torch.tensor(
        [1.0] * src_cite.numel(), device="cuda"
    )
    feature_store[("author", "writes", "paper"), "bias", None] = torch.tensor(
        [1.0] * src_author.numel(), device="cuda"
    )

    loader = cugraph_pyg.loader.LinkNeighborLoader(
        (feature_store, graph_store),
        num_neighbors={
            ("paper", "cites", "paper"): [2, 2, 2],
            ("author", "writes", "paper"): [2, 2, 0],
        },
        batch_size=1,
        edge_label_index=(("author", "writes", "paper"), torch.tensor([[0], [3]])),
        edge_label_time=torch.tensor([-1]),
        time_attr="time",
        weight_attr="bias" if biased else None,
        shuffle=False,
        temporal_comparison="strictly_increasing",
    )

    out = next(iter(loader))

    assert sorted(out["author"].n_id.tolist()) == [0, 1, 2]
    assert out["paper"].n_id.tolist() == [3, 2, 1, 0]

    assert sorted(out["author", "writes", "paper"].e_id.tolist()) == [0, 2, 4, 5]
    assert out["author", "writes", "paper"].num_sampled_edges.tolist() == [2, 2, 0]

    # FIXME resolve issues with num_sampled_nodes


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("neg_sampling_mode", ["binary", "triplet"])
@pytest.mark.sg
def test_link_neighbor_loader_temporal_negative_sampling_homogeneous(
    single_pytorch_worker, batch_size, neg_sampling_mode
):
    """
    Test temporal negative sampling for homogeneous graphs.
    This test ensures that:
    1. Negative samples are generated respecting temporal constraints
    2. Negative sample nodes have timestamps <= edge_label_time
    3. Both positive and negative samples are present in the output
    """
    # Create a homogeneous temporal graph with paper-paper citations
    src_cite = torch.tensor([3, 2, 1, 2, 3, 4, 0])  # paper
    dst_cite = torch.tensor([2, 1, 0, 0, 1, 2, 1])  # paper
    tme_cite = torch.tensor([5, 6, 7, 3, 4, 8, 2])  # edge timestamps

    num_papers = 5

    # Create node timestamps (papers are created/published at specific times)
    node_time = torch.tensor([0, 1, 2, 3, 4])  # paper timestamps

    graph_store = GraphStore()
    feature_store = FeatureStore()

    # Add paper-paper citations
    graph_store[("paper", "cites", "paper"), "coo", False, (num_papers, num_papers)] = [
        dst_cite,
        src_cite,
    ]

    # Add time attributes for both edges and nodes
    feature_store[("paper", "cites", "paper"), "time", None] = tme_cite
    feature_store["paper", "time", None] = node_time

    # Create edge label index for link prediction
    # We'll test prediction for a subset of edges
    edge_label_index = torch.tensor([[3, 2], [2, 1]])
    edge_label_time = torch.tensor([10, 10])  # Future time for prediction

    # Configure negative sampling
    if neg_sampling_mode == "binary":
        neg_sampling = torch_geometric.sampler.NegativeSampling("binary", amount=2.0)
    else:
        neg_sampling = torch_geometric.sampler.NegativeSampling("triplet", amount=2.0)

    loader = cugraph_pyg.loader.LinkNeighborLoader(
        (feature_store, graph_store),
        num_neighbors=[2, 2],
        batch_size=batch_size,
        edge_label_index=edge_label_index,
        edge_label_time=edge_label_time,
        time_attr="time",
        neg_sampling=neg_sampling,
        shuffle=False,
    )

    # Test that the loader produces batches with proper temporal negative sampling
    total_pos = total_neg = 0
    for i, batch in enumerate(loader):
        # Check that we have edge labels
        assert hasattr(batch, "edge_label")
        assert hasattr(batch, "edge_label_index")

        # Should have both positive (1.0) and negative (0.0) labels
        edge_labels = batch.edge_label
        assert torch.any(edge_labels == 1.0), "Should have positive labels"
        assert torch.any(edge_labels == 0.0), "Should have negative labels"

        # Verify negative sampling ratio
        num_pos = (edge_labels == 1.0).sum().item()
        num_neg = (edge_labels == 0.0).sum().item()

        total_pos += num_pos
        total_neg += num_neg

        # Verify that edge label index has the correct shape
        edge_label_idx = batch.edge_label_index
        assert edge_label_idx.shape[0] == 2
        assert edge_label_idx.shape[1] == len(edge_labels)

        # For temporal negative sampling, verify that the negative sample nodes
        # have timestamps that respect temporal constraints
        # Negative samples should only use nodes that existed before edge_label_time
        neg_mask = edge_labels == 0.0
        if neg_mask.sum() > 0:
            neg_src = edge_label_idx[0, neg_mask]
            neg_dst = edge_label_idx[1, neg_mask]

            # Convert to original node IDs and check their timestamps
            src_node_ids = batch.n_id[neg_src.cpu()].cpu()
            dst_node_ids = batch.n_id[neg_dst.cpu()].cpu()

            # All negative sample nodes should have timestamps <= edge_label_time
            # for the corresponding batch element
            for j in range(len(neg_src)):
                src_id = src_node_ids[j].item()
                dst_id = dst_node_ids[j].item()
                # Node timestamps should be <= edge prediction time
                assert node_time[src_id] <= edge_label_time[i * batch_size].item()
                assert node_time[dst_id] <= edge_label_time[i * batch_size].item()

    assert total_neg == 2 * total_pos, (
        f"Expected 2:1 negative:positive ratio, got {num_neg}:{num_pos}"
    )

    # Verify we processed all batches
    assert i >= 0


@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("neg_sampling_mode", ["binary", "triplet"])
@pytest.mark.sg
def test_link_neighbor_loader_temporal_negative_sampling_heterogeneous(
    single_pytorch_worker, batch_size, neg_sampling_mode
):
    """
    Test temporal negative sampling for heterogeneous graphs.
    This test ensures that temporal constraints are properly enforced
    across different node and edge types.
    """
    # Create a heterogeneous graph with papers, authors, and citations
    src_cite = torch.tensor([3, 2, 1, 2])  # paper
    dst_cite = torch.tensor([2, 1, 0, 0])  # paper
    tme_cite = torch.tensor([5, 6, 7, 3])  # edge timestamps

    src_author = torch.tensor([3, 2, 2, 1, 3, 2, 0])  # paper
    dst_author = torch.tensor([0, 0, 1, 1, 2, 2, 2])  # author
    tme_author = torch.tensor([4, 3, 5, 2, 7, 6, 1])  # edge timestamps

    num_papers = 4
    num_authors = 3

    # Create node timestamps
    paper_time = torch.tensor([0, 1, 2, 3])  # paper timestamps
    author_time = torch.tensor([0, 1, 2])  # author timestamps

    graph_store = GraphStore()
    feature_store = FeatureStore()

    # Add edges
    graph_store[("paper", "cites", "paper"), "coo", False, (num_papers, num_papers)] = [
        dst_cite,
        src_cite,
    ]
    graph_store[
        ("author", "writes", "paper"), "coo", False, (num_authors, num_papers)
    ] = [
        dst_author,
        src_author,
    ]

    # Add time attributes for edges
    feature_store[("paper", "cites", "paper"), "time", None] = tme_cite
    feature_store[("author", "writes", "paper"), "time", None] = tme_author

    # Add time attributes for nodes
    feature_store["paper", "time", None] = paper_time
    feature_store["author", "time", None] = author_time

    # Create edge label index for author-paper relationships
    edge_label_index = torch.stack(
        [
            torch.tensor([0, 1, 2]),  # authors
            torch.tensor([3, 2, 1]),  # papers
        ]
    )
    edge_label_time = torch.tensor([8, 8, 8])  # Future time for prediction

    # Configure negative sampling
    if neg_sampling_mode == "binary":
        neg_sampling = torch_geometric.sampler.NegativeSampling("binary", amount=2.0)
    else:
        neg_sampling = torch_geometric.sampler.NegativeSampling("triplet", amount=2.0)

    loader = cugraph_pyg.loader.LinkNeighborLoader(
        (feature_store, graph_store),
        num_neighbors={
            ("paper", "cites", "paper"): [2, 2],
            ("author", "writes", "paper"): [2, 2],
        },
        batch_size=batch_size,
        edge_label_index=(("author", "writes", "paper"), edge_label_index),
        edge_label_time=edge_label_time,
        time_attr="time",
        neg_sampling=neg_sampling,
        shuffle=False,
    )

    # Test that the loader produces batches with proper temporal negative sampling
    total_pos = total_neg = 0
    for i, batch in enumerate(loader):
        # Check that we have the expected edge types
        assert "author" in batch.node_types
        assert "paper" in batch.node_types

        # Check edge label structure
        assert [("author", "writes", "paper")] == list(
            batch.edge_label_index_dict.keys()
        )
        assert [("author", "writes", "paper")] == list(batch.edge_label_dict.keys())

        # Should have both positive and negative labels
        edge_labels = batch["author", "writes", "paper"].edge_label
        assert torch.any(edge_labels == 1.0), "Should have positive labels"
        assert torch.any(edge_labels == 0.0), "Should have negative labels"

        # Verify negative sampling ratio
        num_pos = (edge_labels == 1.0).sum().item()
        num_neg = (edge_labels == 0.0).sum().item()
        total_pos += num_pos
        total_neg += num_neg

        # Verify edge label index shape
        edge_label_idx = batch["author", "writes", "paper"].edge_label_index
        assert edge_label_idx.shape[0] == 2
        assert edge_label_idx.shape[1] == len(edge_labels)

        # Verify temporal constraints for negative samples
        neg_mask = edge_labels == 0.0
        if neg_mask.sum() > 0:
            neg_src = edge_label_idx[0, neg_mask]
            neg_dst = edge_label_idx[1, neg_mask]

            # Convert to original node IDs
            author_node_ids = batch["author"].n_id[neg_src.cpu()].cpu()
            paper_node_ids = batch["paper"].n_id[neg_dst.cpu()].cpu()

            # All negative sample nodes should have timestamps <= edge_label_time
            for j in range(len(neg_src)):
                author_id = author_node_ids[j].item()
                paper_id = paper_node_ids[j].item()
                # Node timestamps should be <= edge prediction time
                assert author_time[author_id] <= edge_label_time[i * batch_size].item()
                assert paper_time[paper_id] <= edge_label_time[i * batch_size].item()

        # Verify that node IDs are valid
        assert batch["author"].n_id.numel() > 0
        assert batch["paper"].n_id.numel() > 0

    assert total_neg == 2 * total_pos, (
        f"Expected 2:1 negative:positive ratio, got {total_neg}:{total_pos}"
    )

    # Verify we processed all batches
    assert i >= 0
