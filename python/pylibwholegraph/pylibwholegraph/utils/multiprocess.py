# SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import multiprocessing as mp


def multiprocess_run(world_size: int, func, inline_single_process=False):
    """
    Run func in multiple process
    :param world_size: process count
    :param func: function to run
    :param inline_single_process: when only one process,
      whether to use current process to run.
    :return: None
    """
    assert world_size > 0
    if world_size == 1 and inline_single_process:
        func(0, 1)
        return
    spawn_context = mp.get_context("spawn")

    process_array = [None] * world_size
    for i in range(world_size):
        process_array[i] = spawn_context.Process(target=func, args=(i, world_size))
        process_array[i].start()
    for i in range(world_size):
        process_array[i].join()
    for i in range(world_size):
        assert process_array[i].exitcode == 0
