# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import rmm

import pylibwholegraph.binding.wholememory_binding as wmb
import pylibwholegraph.torch as wgth


@pytest.fixture
def restore_memory_resource():
    previous_resource = rmm.mr.get_current_device_resource()
    wmb.set_rmm_enabled(False)
    try:
        yield
    finally:
        wmb.set_rmm_enabled(False)
        rmm.mr.set_current_device_resource(previous_resource)


def test_set_rmm_enabled_uses_current_resource(restore_memory_resource):
    memory_resource = rmm.mr.CudaMemoryResource()
    rmm.mr.set_current_device_resource(memory_resource)
    wgth.set_rmm_enabled(True)

    assert wgth.is_rmm_enabled()
    assert rmm.mr.get_current_device_resource() is memory_resource


def test_finalize_disables_rmm(restore_memory_resource):
    wmb.init(0)
    current_resource = rmm.mr.get_current_device_resource()
    wgth.set_rmm_enabled(True)

    wgth.finalize()

    assert not wgth.is_rmm_enabled()
    assert rmm.mr.get_current_device_resource() is current_resource
