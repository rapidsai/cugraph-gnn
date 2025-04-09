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

include(../../rapids_config.cmake)
# gets all the other stuff rapids exports, etc
include(rapids-cmake)
# cmds for configuring cuda projects
include(rapids-cuda)
# cmake package manager does the get_*.cmake stuff to search for and maybe pull in sources
include(rapids-cpm)

# from rapids-cuda setting rapids_cuda_architectures(ish) passed to nvcc to specify CUDA
# architectures to build for (e.g. 70-virtual, 80-real, etc)
rapids_cuda_init_architectures(LIBWHOLEGRAPH)

# when you do this cmake sets a bunch of variables in scope libwholegraph_version
# libwhoelgraph_majorversion etc...
project(
  LIBWHOLEGRAPH
  VERSION "${RAPIDS_VERSION}"
  # searching for a working ___ compiler... in cmake logs sets up compiler flags
  LANGUAGES CXX CUDA
)

rapids_cpm_init()

# * Dependencies

# set these before we call add_subdirectory (where they are defined with default values)
SET(BUILD_TESTS OFF)
SET(BUILD_BENCHMARKS OFF)

add_subdirectory(../../cpp/ libwholegraph)

message(STATUS "LIBWHOLEGRAPH: building statically linked wheel")
