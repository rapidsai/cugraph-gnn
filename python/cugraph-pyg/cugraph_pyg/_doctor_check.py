# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Smoke check for `rapids doctor` (RAPIDS CLI).

See: https://github.com/rapidsai/rapids-cli#check-plugins
"""


def cugraph_pyg_smoke_check(**kwargs):
    """
    A quick check to ensure cugraph-pyg can be imported and its core
    submodules are loadable.
    """
    try:
        import cugraph_pyg

        # Ensure core submodules load (touches pylibwholegraph, torch-geometric, etc.)
        import cugraph_pyg.data
        import cugraph_pyg.tensor

    except ImportError as e:
        raise ImportError(
            "cugraph-pyg or its dependencies could not be imported. "
            "Tip: install with `pip install cugraph-pyg` or use a RAPIDS conda environment."
        ) from e

    if not hasattr(cugraph_pyg, "__version__") or not cugraph_pyg.__version__:
        raise AssertionError(
            "cugraph-pyg smoke check failed: __version__ not found or empty"
        )

    from cugraph_pyg.utils.imports import import_optional, MissingModule

    torch = import_optional("torch")

    if isinstance(torch, MissingModule):
        import warnings

        warnings.warn(
            "PyTorch is required to use cuGraph-PyG."
            "Please install PyTorch from PyPI or Conda-Forge."
        )
    else:
        import os
        from cugraph_pyg.data import GraphStore

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29505"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        torch.distributed.init_process_group("nccl")

        graph_store = GraphStore()
        graph_store.put_edge_index(
            torch.tensor([[0, 1], [1, 2]]),
            ("person", "knows", "person"),
            "coo",
            False,
            (3, 3),
        )
        edge_index = graph_store.get_edge_index(("person", "knows", "person"), "coo")
        assert edge_index.shape == torch.Size([2, 2])

        torch.distributed.destroy_process_group()
