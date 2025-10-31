# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cuda)

rapids_cuda_init_architectures(LIBWHOLEGRAPH)

project(
  LIBWHOLEGRAPH
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

# * Dependencies

SET(BUILD_TESTS OFF)
SET(BUILD_BENCHMARKS OFF)
SET(USE_NCCL_RUNTIME_WHEEL ON)

add_subdirectory(../../cpp/ libwholegraph)

if(USE_NCCL_RUNTIME_WHEEL)
  list(APPEND rpaths "$ORIGIN/../../nvidia/nccl/lib")
endif()

set_property(
  TARGET wholegraph
  PROPERTY INSTALL_RPATH ${rpaths}
  APPEND
)
