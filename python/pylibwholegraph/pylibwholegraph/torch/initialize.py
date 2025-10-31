# SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torch.utils.dlpack
import pylibwholegraph.binding.wholememory_binding as wmb
from .comm import (
    set_world_info,
    get_global_communicator,
    get_local_node_communicator,
    reset_communicators,
)
from .utils import str_to_wmb_wholememory_log_level


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
