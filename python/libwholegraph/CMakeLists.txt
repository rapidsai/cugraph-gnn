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
include(rapids-cmake)
include(rapids-cuda)
include(rapids-cpm)

rapids_cuda_init_architectures(LIBWHOLEGRAPH)

project(
  LIBWHOLEGRAPH
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

# default build type
rapids_cmake_build_type(Release)

rapids_cpm_init()

include(../../cpp/cmake/thirdparty/get_raft.cmake)

# ##################################################################################################
# * Dependencies  ------------------------------------------------------------

message(STATUS "LIBWHOLEGRAPH: building statically linked wheel")
add_subdirectory(../../cpp/ libwholegraph EXCLUDE_FROM_ALL)
