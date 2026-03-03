# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Smoke check for `rapids doctor` (RAPIDS CLI).

See: https://github.com/rapidsai/rapids-cli#check-plugins
"""


def pylibwholegraph_smoke_check(**kwargs):
    """
    A quick check to ensure pylibwholegraph can be imported and the
    native library loads correctly.
    """
    try:
        import pylibwholegraph
    except ImportError as e:
        raise ImportError(
            "pylibwholegraph or its dependencies could not be imported. "
            "Tip: install with `pip install pylibwholegraph` or use a RAPIDS conda environment."
        ) from e

    if not hasattr(pylibwholegraph, "__version__") or not pylibwholegraph.__version__:
        raise AssertionError(
            "pylibwholegraph smoke check failed: __version__ not found or empty"
        )
