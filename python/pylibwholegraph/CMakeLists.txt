# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

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

# Find CMake packages provided by installed wheels.
find_package(Python COMPONENTS Interpreter)
if(Python_FOUND)
  include(${rapids-cmake-dir}/cython-core/find_prefix_paths.cmake)
  rapids_cython_find_prefix_paths("${Python_EXECUTABLE}" wheel_prefix_paths)
  set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE)
  list(PREPEND CMAKE_PREFIX_PATH ${wheel_prefix_paths})
endif()

find_package(wholegraph "${RAPIDS_VERSION}" REQUIRED)

include(rapids-cython-core)
rapids_cython_init()

# ##################################################################################################
# * Cython modules -----------------------------------------------------------

add_subdirectory(pylibwholegraph/binding)
