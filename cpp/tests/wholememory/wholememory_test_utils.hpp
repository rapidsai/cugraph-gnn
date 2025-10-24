/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <functional>
#include <thread>

#include "parallel_utils.hpp"
#include "wholememory/communicator.hpp"

wholememory_comm_t create_communicator_by_pipes(const std::vector<std::array<int, 2>>& pipes,
                                                int rank,
                                                int world_size)
{
  wholememory_unique_id_t unique_id;
  if (rank == 0) {
    WHOLEMEMORY_CHECK_NOTHROW(wholememory::create_unique_id(&unique_id) == WHOLEMEMORY_SUCCESS);
  }

  PipeBroadcast(rank, world_size, 0, pipes, &unique_id);

  wholememory_comm_t wm_comm;
  WHOLEMEMORY_CHECK_NOTHROW(
    wholememory::create_communicator(&wm_comm, unique_id, rank, world_size) == WHOLEMEMORY_SUCCESS);
  return wm_comm;
}

wholememory_comm_t create_group_communicator_by_pipes(const std::vector<std::array<int, 2>>& pipes,
                                                      int rank,
                                                      int world_size,
                                                      int group_count)
{
  WHOLEMEMORY_CHECK_NOTHROW(world_size % group_count == 0);
  int group_size = world_size / group_count;
  int group_rank = rank % group_size;
  wholememory_unique_id_t unique_id;
  if (group_rank == 0) {
    WHOLEMEMORY_CHECK_NOTHROW(wholememory::create_unique_id(&unique_id) == WHOLEMEMORY_SUCCESS);
  }

  wholememory_unique_id_t comm_unique_id;
  for (int g = 0; g < group_count; g++) {
    if (g * group_size == rank) comm_unique_id = unique_id;
    PipeBroadcast(rank, world_size, g * group_size, pipes, &comm_unique_id);
    if (rank / group_size == g) unique_id = comm_unique_id;
  }

  wholememory_comm_t wm_comm;
  WHOLEMEMORY_CHECK_NOTHROW(wholememory::create_communicator(
                              &wm_comm, unique_id, group_rank, group_size) == WHOLEMEMORY_SUCCESS);
  return wm_comm;
}

wholememory_comm_t create_communicator_by_socket(SideBandCommunicator* side_band_communicator,
                                                 int rank,
                                                 int world_size)
{
  wholememory_unique_id_t unique_id;
  if (rank == 0) {
    WHOLEMEMORY_CHECK_NOTHROW(wholememory::create_unique_id(&unique_id) == WHOLEMEMORY_SUCCESS);
  }

  SideBandBroadcast(side_band_communicator, &unique_id, sizeof(wholememory_unique_id_t), 0);

  wholememory_comm_t wm_comm;
  WHOLEMEMORY_CHECK_NOTHROW(
    wholememory::create_communicator(&wm_comm, unique_id, rank, world_size) == WHOLEMEMORY_SUCCESS);
  return wm_comm;
}
