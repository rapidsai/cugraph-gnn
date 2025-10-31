/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>

#include "parallel_utils.hpp"
#include "wholememory/communicator.hpp"

#include "wholememory_test_utils.hpp"

TEST(WholeMemoryCommTest, SimpleCreateDestroyCommunicator)
{
  int dev_count = ForkGetDeviceCount();
  EXPECT_GE(dev_count, 1);
  WHOLEMEMORY_CHECK(dev_count >= 1);
  int nproc = dev_count;
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, dev_count);
  MultiProcessRun(nproc, [&pipes](int rank, int world_size) {
    EXPECT_EQ(cudaSetDevice(rank), cudaSuccess);
    wholememory_comm_t wm_comm1 = create_communicator_by_pipes(pipes, rank, world_size);
    EXPECT_EQ(wm_comm1->comm_id, 0);
    EXPECT_EQ(wholememory::destroy_all_communicators(), WHOLEMEMORY_SUCCESS);

    WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
  });
}

TEST(WholeMemoryCommTest, CommunicatorFunctions)
{
  int dev_count = ForkGetDeviceCount();
  EXPECT_GE(dev_count, 1);
  WHOLEMEMORY_CHECK(dev_count >= 1);
  int nproc = dev_count;
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, dev_count);
  MultiProcessRun(nproc, [&pipes](int rank, int world_size) {
    EXPECT_EQ(cudaSetDevice(rank), cudaSuccess);
    wholememory_comm_t wm_comm1 = create_communicator_by_pipes(pipes, rank, world_size);
    EXPECT_EQ(wm_comm1->comm_id, 0);
    int comm_rank = -1;
    EXPECT_EQ(wholememory::communicator_get_rank(&comm_rank, wm_comm1), WHOLEMEMORY_SUCCESS);
    EXPECT_EQ(comm_rank, rank);
    int comm_size = 0;
    EXPECT_EQ(wholememory::communicator_get_size(&comm_size, wm_comm1), WHOLEMEMORY_SUCCESS);
    EXPECT_EQ(comm_size, world_size);
    EXPECT_EQ(wholememory::is_intranode_communicator(wm_comm1), true);
    EXPECT_EQ(wholememory::destroy_all_communicators(), WHOLEMEMORY_SUCCESS);

    WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
  });
}

TEST(WholeMemoryCommTest, MultipleCreateDestroyCommunicator)
{
  int dev_count = ForkGetDeviceCount();
  EXPECT_GE(dev_count, 1);
  WHOLEMEMORY_CHECK(dev_count >= 1);
  int nproc = dev_count;
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, dev_count);
  MultiProcessRun(nproc, [&pipes](int rank, int world_size) {
    EXPECT_EQ(cudaSetDevice(rank), cudaSuccess);
    wholememory_comm_t wm_comm1 = create_communicator_by_pipes(pipes, rank, world_size);
    EXPECT_EQ(wm_comm1->comm_id, 0);
    wholememory_comm_t wm_comm2 = create_communicator_by_pipes(pipes, rank, world_size);
    EXPECT_EQ(wm_comm2->comm_id, 1);
    EXPECT_EQ(wholememory::destroy_communicator(wm_comm1), WHOLEMEMORY_SUCCESS);
    wholememory_comm_t wm_comm3 = create_communicator_by_pipes(pipes, rank, world_size);
    EXPECT_EQ(wm_comm3->comm_id, 0);
    EXPECT_EQ(wholememory::destroy_all_communicators(), WHOLEMEMORY_SUCCESS);
    wholememory_comm_t wm_comm4 = create_communicator_by_pipes(pipes, rank, world_size);
    EXPECT_EQ(wm_comm4->comm_id, 0);
    EXPECT_EQ(wholememory::destroy_all_communicators(), WHOLEMEMORY_SUCCESS);

    WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
  });
}
