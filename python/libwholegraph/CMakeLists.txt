# =============================================================================
# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
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
