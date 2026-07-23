# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pylibwholegraph.utils.imports import import_optional
import pylibwholegraph.binding.wholememory_binding as wmb
from .comm import (
    set_world_info,
    get_global_communicator,
    get_local_node_communicator,
    reset_communicators,
)
from .utils import str_to_wmb_wholememory_log_level

torch = import_optional("torch")
_memory_resources = []


def set_memory_resource(memory_resource):
    r"""Set the RMM memory resource used by supported WholeMemory allocations.

    The resource is installed as RMM's current resource for the active CUDA device. Future
    distributed and hierarchy WholeMemory tensors with device storage use this resource. Chunked,
    continuous, and NVSHMEM device tensors retain their specialized allocation paths and emit a
    warning when RMM is enabled.

    Pass ``None`` to disable RMM for future WholeMemory allocations. Existing allocations retain
    the allocator with which they were created.

    This function must be called once per process after selecting the process's CUDA device and
    before creating WholeMemory tensors.

    :param memory_resource: An ``rmm.mr.DeviceMemoryResource``, or ``None`` to disable RMM.
    :return: None
    """
    global _memory_resources
    if memory_resource is None:
        wmb.set_rmm_enabled(False)
        return

    try:
        import rmm
    except ImportError as exc:
        raise ImportError(
            "Setting a WholeMemory memory resource requires the Python rmm package."
        ) from exc

    if not isinstance(memory_resource, rmm.mr.DeviceMemoryResource):
        raise TypeError("memory_resource must be an rmm.mr.DeviceMemoryResource")

    rmm.mr.set_current_device_resource(memory_resource)
    if all(memory_resource is not resource for resource in _memory_resources):
        _memory_resources.append(memory_resource)
    wmb.set_rmm_enabled(True)


def is_rmm_enabled():
    r"""Return whether RMM is enabled for supported WholeMemory device allocations."""
    return wmb.is_rmm_enabled()


def init(
    world_rank: int,
    world_size: int,
    local_rank: int,
    local_size: int,
    wm_log_level="info",
):
    wmb.init(0, str_to_wmb_wholememory_log_level(wm_log_level))
    set_world_info(world_rank, world_size, local_rank, local_size)


def init_torch_env(
    world_rank: int,
    world_size: int,
    local_rank: int,
    local_size: int,
    wm_log_level="info",
):
    r"""Init WholeGraph environment for PyTorch.
    :param world_rank: world rank of current process
    :param world_size: world size of all processes
    :param local_rank: local rank of current process
    :param local_size: local size
    :return: None
    """
    os.environ["RANK"] = str(world_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    if "MASTER_ADDR" not in os.environ:
        if world_rank == 0:
            print("[WARNING] MASTER_ADDR not set, resetting to localhost")
        os.environ["MASTER_ADDR"] = "localhost"

    if "MASTER_PORT" not in os.environ:
        if world_rank == 0:
            print("[WARNING] MASTER_PORT not set, resetting to 12335")
        os.environ["MASTER_PORT"] = "12335"

    wmb.init(0, str_to_wmb_wholememory_log_level(wm_log_level))
    torch.set_num_threads(1)
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    set_world_info(world_rank, world_size, local_rank, local_size)


def init_torch_env_and_create_wm_comm(
    world_rank: int,
    world_size: int,
    local_rank: int,
    local_size: int,
    distributed_backend_type="nccl",
    wm_log_level="info",
):
    r"""Init WholeGraph environment for PyTorch and create
      single communicator for all ranks.
    :param world_rank: world rank of current process
    :param world_size: world size of all processes
    :param local_rank: local rank of current process
    :param local_size: local size
    :return: global and local node Communicator
    """
    init_torch_env(world_rank, world_size, local_rank, local_size, wm_log_level)
    global_comm = get_global_communicator(distributed_backend_type)
    local_comm = get_local_node_communicator()

    return global_comm, local_comm


def finalize():
    r"""Finalize WholeGraph.
    :return: None
    """
    wmb.finalize()
    reset_communicators()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
