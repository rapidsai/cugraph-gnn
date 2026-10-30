# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util

import pytest


@pytest.fixture(scope="function")
def torch():
    """Pass this to any test case that needs 'torch' to be installed"""
    return pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def finalize_wholegraph_state(request):
    """
    Some tests under ``wholegraph_torch/`` run their routine in the pytest process
    itself (``multiprocess_run(..., inline_single_process=True)``), which initializes
    WholeMemory and the default torch process group as process-global state.

    Tear both down after every such test so that a leak (or a failure partway
    through a test) can't break the tests that run after it by leaving that WholeMemory
    state initialized.
    """
    yield

    # only apply to tests in 'wholegraph_torch/'
    if "wholegraph_torch" not in request.path.parts:
        return
    if importlib.util.find_spec("torch") is None:
        return
    from pylibwholegraph.torch.initialize import finalize

    finalize()
