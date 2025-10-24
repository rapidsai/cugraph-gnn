# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 3.30.4 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cuda)
include(rapids-cpm)

rapids_cuda_init_architectures(PYLIBWHOLEGRAPH)

project(
  PYLIBWHOLEGRAPH
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

find_package(wholegraph "${RAPIDS_VERSION}" REQUIRED)

include(rapids-cython-core)
rapids_cython_init()

# ##################################################################################################
# * Cython modules -----------------------------------------------------------

add_subdirectory(pylibwholegraph/binding)
