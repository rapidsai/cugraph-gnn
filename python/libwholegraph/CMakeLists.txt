# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cuda)

rapids_cuda_init_architectures(LIBWHOLEGRAPH)

project(
  LIBWHOLEGRAPH
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

# Find C++ libraries from RAPIDS wheels in site-packages.
set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE)

# Check if wholegraph is already available. If so, it is the user's responsibility to ensure that
# the CMake package is also available at build time of Python packages that need libwholegraph.
find_package(wholegraph "${RAPIDS_VERSION}")

if(wholegraph_FOUND)
  return()
endif()

unset(wholegraph_FOUND)

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
