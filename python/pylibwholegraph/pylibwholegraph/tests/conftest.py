# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture(scope="function")
def torch():
    """Pass this to any test case that needs 'torch' to be installed"""
    return pytest.importorskip("torch")
