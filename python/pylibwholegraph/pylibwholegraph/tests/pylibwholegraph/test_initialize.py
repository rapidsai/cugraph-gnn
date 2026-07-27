# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import rmm

import pylibwholegraph.binding.wholememory_binding as wmb
import pylibwholegraph.torch as wgth
import pylibwholegraph.torch.initialize as initialize


@pytest.fixture
def restore_memory_resource():
    previous_resource = rmm.mr.get_current_device_resource()
    retained_resource_count = len(initialize._memory_resources)
    wmb.set_rmm_enabled(False)
    try:
        yield
    finally:
        wmb.set_rmm_enabled(False)
        rmm.mr.set_current_device_resource(previous_resource)
        del initialize._memory_resources[retained_resource_count:]


def test_set_memory_resource(restore_memory_resource):
    memory_resource = rmm.mr.CudaMemoryResource()

    wgth.set_memory_resource(memory_resource)

    assert wgth.is_rmm_enabled()
    assert rmm.mr.get_current_device_resource() is memory_resource
    assert any(resource is memory_resource for resource in initialize._memory_resources)

    wgth.set_memory_resource(None)

    assert not wgth.is_rmm_enabled()


def test_set_memory_resource_rejects_invalid_type(restore_memory_resource):
    with pytest.raises(
        TypeError, match="memory_resource must be an rmm.mr.DeviceMemoryResource"
    ):
        wgth.set_memory_resource(object())


def test_finalize_disables_rmm(restore_memory_resource):
    wmb.init(0)
    wgth.set_memory_resource(rmm.mr.CudaMemoryResource())

    wgth.finalize()

    assert not wgth.is_rmm_enabled()
