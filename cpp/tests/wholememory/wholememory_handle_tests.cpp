/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>

#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/statistics_resource_adaptor.hpp>

#include <utility>

#include "parallel_utils.hpp"
#include "wholememory/communicator.hpp"
#include "wholememory/initialize.hpp"
#include "wholememory/memory_handle.hpp"

#include "wholememory_test_utils.hpp"

class WholeMemoryHandleCreateDestroyParameterTests
  : public ::testing::TestWithParam<
      std::tuple<size_t, wholememory_memory_type_t, wholememory_memory_location_t, size_t>> {};

TEST_P(WholeMemoryHandleCreateDestroyParameterTests, CreateDestroyTest)
{
  auto params   = GetParam();
  int dev_count = ForkGetDeviceCount();
  EXPECT_GE(dev_count, 1);
  WHOLEMEMORY_CHECK(dev_count >= 1);
  int nproc = dev_count;
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, dev_count);
  MultiProcessRun(
    nproc,
    [&pipes, &params](int rank, int world_size) {
      EXPECT_EQ(wholememory_init(0), WHOLEMEMORY_SUCCESS);

      EXPECT_EQ(cudaSetDevice(rank), cudaSuccess);

      wholememory_comm_t wm_comm = create_communicator_by_pipes(pipes, rank, world_size);

      if (wholememory_communicator_support_type_location(
            wm_comm, std::get<1>(params), std::get<2>(params)) != WHOLEMEMORY_SUCCESS) {
        EXPECT_EQ(wholememory::destroy_all_communicators(), WHOLEMEMORY_SUCCESS);
        EXPECT_EQ(wholememory_finalize(), WHOLEMEMORY_SUCCESS);
        WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
        if (rank == 0) GTEST_SKIP_("Skip due to not supported.");
        return;
      }

      wholememory_handle_t handle1;
      EXPECT_EQ(wholememory::create_wholememory(&handle1,
                                                std::get<0>(params),
                                                wm_comm,
                                                std::get<1>(params),
                                                std::get<2>(params),
                                                std::get<2>(params)),
                WHOLEMEMORY_SUCCESS);

      EXPECT_EQ(wholememory::destroy_wholememory(handle1), WHOLEMEMORY_SUCCESS);

      EXPECT_EQ(wholememory::destroy_all_communicators(), WHOLEMEMORY_SUCCESS);

      EXPECT_EQ(wholememory_finalize(), WHOLEMEMORY_SUCCESS);

      WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
    },
    true);
  ClosePipes(&pipes);
}

INSTANTIATE_TEST_SUITE_P(
  WholeMemoryHandleTests,
  WholeMemoryHandleCreateDestroyParameterTests,
  ::testing::Values(
    std::make_tuple(
      1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_CONTINUOUS, WHOLEMEMORY_ML_DEVICE, 128UL),
    std::make_tuple(1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_CONTINUOUS, WHOLEMEMORY_ML_HOST, 128UL),
    std::make_tuple(1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_CHUNKED, WHOLEMEMORY_ML_DEVICE, 128UL),
    std::make_tuple(1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_CHUNKED, WHOLEMEMORY_ML_HOST, 128UL),
    std::make_tuple(
      1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_DISTRIBUTED, WHOLEMEMORY_ML_DEVICE, 128UL),
    std::make_tuple(
      1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_DISTRIBUTED, WHOLEMEMORY_ML_HOST, 128UL),

    std::make_tuple(
      1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_CONTINUOUS, WHOLEMEMORY_ML_DEVICE, 63UL),
    std::make_tuple(1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_CONTINUOUS, WHOLEMEMORY_ML_HOST, 63UL),
    std::make_tuple(1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_CHUNKED, WHOLEMEMORY_ML_DEVICE, 63UL),
    std::make_tuple(1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_CHUNKED, WHOLEMEMORY_ML_HOST, 63UL),
    std::make_tuple(
      1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_DISTRIBUTED, WHOLEMEMORY_ML_DEVICE, 63UL),
    std::make_tuple(1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_DISTRIBUTED, WHOLEMEMORY_ML_HOST, 63UL),

    std::make_tuple(
      1024UL * 1024UL * 512UL, WHOLEMEMORY_MT_CONTINUOUS, WHOLEMEMORY_ML_HOST, 128UL)));

class WholeMemoryHandleMultiCreateParameterTests
  : public ::testing::TestWithParam<
      std::tuple<wholememory_memory_type_t, wholememory_memory_location_t>> {};

TEST_P(WholeMemoryHandleMultiCreateParameterTests, CreateDestroyTest)
{
  auto params   = GetParam();
  int dev_count = ForkGetDeviceCount();
  EXPECT_GE(dev_count, 1);
  WHOLEMEMORY_CHECK(dev_count >= 1);
  int nproc = dev_count;
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, dev_count);
  MultiProcessRun(
    nproc,
    [&pipes, &params](int rank, int world_size) {
      EXPECT_EQ(wholememory_init(0), WHOLEMEMORY_SUCCESS);

      EXPECT_EQ(cudaSetDevice(rank), cudaSuccess);

      wholememory_comm_t wm_comm = create_communicator_by_pipes(pipes, rank, world_size);

      if (wholememory_communicator_support_type_location(
            wm_comm, std::get<0>(params), std::get<1>(params)) != WHOLEMEMORY_SUCCESS) {
        EXPECT_EQ(wholememory::destroy_all_communicators(), WHOLEMEMORY_SUCCESS);
        EXPECT_EQ(wholememory_finalize(), WHOLEMEMORY_SUCCESS);
        WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
        if (rank == 0) GTEST_SKIP_("Skip due to not supported.");
        return;
      }

      size_t total_size  = 1024UL * 1024UL * 32;
      size_t granularity = 128;

      wholememory_handle_t handle1, handle2, handle3, handle4, handle5;
      EXPECT_EQ(
        wholememory::create_wholememory(
          &handle1, total_size, wm_comm, std::get<0>(params), std::get<1>(params), granularity),
        WHOLEMEMORY_SUCCESS);
      // handle1: 0
      EXPECT_EQ(handle1->handle_id, 0);

      EXPECT_EQ(
        wholememory::create_wholememory(
          &handle2, total_size, wm_comm, std::get<0>(params), std::get<1>(params), granularity),
        WHOLEMEMORY_SUCCESS);
      // handle1: 0, handle2: 1
      EXPECT_EQ(handle2->handle_id, 1);

      EXPECT_EQ(
        wholememory::create_wholememory(
          &handle3, total_size, wm_comm, std::get<0>(params), std::get<1>(params), granularity),
        WHOLEMEMORY_SUCCESS);
      // handle1: 0, handle2: 1, handle3: 2
      EXPECT_EQ(handle3->handle_id, 2);
      EXPECT_EQ(wm_comm->wholememory_map.size(), 3);

      EXPECT_EQ(wholememory::destroy_wholememory(handle2), WHOLEMEMORY_SUCCESS);
      // handle1: 0, handle3: 2
      EXPECT_EQ(wm_comm->wholememory_map.size(), 2);

      EXPECT_EQ(
        wholememory::create_wholememory(
          &handle4, total_size, wm_comm, std::get<0>(params), std::get<1>(params), granularity),
        WHOLEMEMORY_SUCCESS);
      // handle1: 0, handle4: 1, handle3: 2
      EXPECT_EQ(handle4->handle_id, 1);

      EXPECT_EQ(wholememory::destroy_wholememory(handle1), WHOLEMEMORY_SUCCESS);
      // handle4: 1, handle3: 2
      EXPECT_EQ(wm_comm->wholememory_map.size(), 2);

      EXPECT_EQ(wholememory::destroy_wholememory(handle3), WHOLEMEMORY_SUCCESS);
      // handle4: 1
      EXPECT_EQ(wm_comm->wholememory_map.size(), 1);

      EXPECT_EQ(
        wholememory::create_wholememory(
          &handle5, total_size, wm_comm, std::get<0>(params), std::get<1>(params), granularity),
        WHOLEMEMORY_SUCCESS);
      // handle5: 0, handle4: 1
      EXPECT_EQ(handle5->handle_id, 0);

      EXPECT_EQ(wholememory::destroy_all_communicators(), WHOLEMEMORY_SUCCESS);

      EXPECT_EQ(wholememory_finalize(), WHOLEMEMORY_SUCCESS);

      WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
    },
    true);
  ClosePipes(&pipes);
}

#if 1
INSTANTIATE_TEST_SUITE_P(
  WholeMemoryHandleTests,
  WholeMemoryHandleMultiCreateParameterTests,
  ::testing::Values(std::make_tuple(WHOLEMEMORY_MT_CONTINUOUS, WHOLEMEMORY_ML_HOST),
                    std::make_tuple(WHOLEMEMORY_MT_CONTINUOUS, WHOLEMEMORY_ML_DEVICE),
                    std::make_tuple(WHOLEMEMORY_MT_CHUNKED, WHOLEMEMORY_ML_HOST),
                    std::make_tuple(WHOLEMEMORY_MT_CHUNKED, WHOLEMEMORY_ML_DEVICE),
                    std::make_tuple(WHOLEMEMORY_MT_DISTRIBUTED, WHOLEMEMORY_ML_HOST),
                    std::make_tuple(WHOLEMEMORY_MT_DISTRIBUTED, WHOLEMEMORY_ML_DEVICE)));
#endif

TEST(WholeMemoryHandleRMMTests, UsesRMMForSupportedDeviceMemory)
{
  int dev_count = ForkGetDeviceCount();
  EXPECT_GE(dev_count, 1);
  WHOLEMEMORY_CHECK(dev_count >= 1);
  std::vector<std::array<int, 2>> pipes;
  CreatePipes(&pipes, dev_count);
  MultiProcessRun(
    dev_count,
    [&pipes](int rank, int world_size) {
      EXPECT_EQ(wholememory_init(0), WHOLEMEMORY_SUCCESS);
      EXPECT_EQ(cudaSetDevice(rank), cudaSuccess);

      wholememory_comm_t wm_comm = create_communicator_by_pipes(pipes, rank, world_size);
      rmm::mr::statistics_resource_adaptor statistics_mr{
        rmm::mr::get_current_device_resource_ref()};
      auto previous_mr =
        rmm::mr::set_current_device_resource(rmm::device_async_resource_ref{statistics_mr});

      EXPECT_EQ(wholememory_set_rmm_enabled(true), WHOLEMEMORY_SUCCESS);
      EXPECT_TRUE(wholememory_is_rmm_enabled());

      size_t constexpr total_size{1024UL * 1024UL};
      size_t constexpr granularity{128UL};

      for (auto memory_type : {WHOLEMEMORY_MT_DISTRIBUTED, WHOLEMEMORY_MT_HIERARCHY}) {
        auto const allocations_before = statistics_mr.get_allocations_counter().total;
        auto const bytes_before       = statistics_mr.get_bytes_counter().value;
        wholememory_handle_t handle;
        EXPECT_EQ(wholememory::create_wholememory(
                    &handle, total_size, wm_comm, memory_type, WHOLEMEMORY_ML_DEVICE, granularity),
                  WHOLEMEMORY_SUCCESS);
        EXPECT_GT(statistics_mr.get_allocations_counter().total, allocations_before);
        EXPECT_GT(statistics_mr.get_bytes_counter().value, bytes_before);
        EXPECT_EQ(wholememory::destroy_wholememory(handle), WHOLEMEMORY_SUCCESS);
        EXPECT_EQ(statistics_mr.get_bytes_counter().value, bytes_before);
      }

      auto const fallback_allocations_before = statistics_mr.get_allocations_counter().total;
      wholememory_handle_t chunked_handle;
      EXPECT_EQ(wholememory::create_wholememory(&chunked_handle,
                                                total_size,
                                                wm_comm,
                                                WHOLEMEMORY_MT_CHUNKED,
                                                WHOLEMEMORY_ML_DEVICE,
                                                granularity),
                WHOLEMEMORY_SUCCESS);
      EXPECT_EQ(statistics_mr.get_allocations_counter().total, fallback_allocations_before);
      EXPECT_EQ(wholememory::destroy_wholememory(chunked_handle), WHOLEMEMORY_SUCCESS);

      EXPECT_EQ(wholememory_set_rmm_enabled(false), WHOLEMEMORY_SUCCESS);
      EXPECT_FALSE(wholememory_is_rmm_enabled());
      rmm::mr::set_current_device_resource(std::move(previous_mr));

      EXPECT_EQ(wholememory::destroy_all_communicators(), WHOLEMEMORY_SUCCESS);
      EXPECT_EQ(wholememory_finalize(), WHOLEMEMORY_SUCCESS);
      WHOLEMEMORY_CHECK(::testing::Test::HasFailure() == false);
    },
    true);
  ClosePipes(&pipes);
}
