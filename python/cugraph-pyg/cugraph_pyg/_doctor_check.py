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

    from cugraph_pyg.utils import import_optional, MissingModule

    torch = import_optional("torch")

    if isinstance(torch, MissingModule):
        import warnings

        warnings.warn(
            "PyTorch is required to use cuGraph-PyG."
            "Please install PyTorch from PyPI or Conda-Forge."
        )
