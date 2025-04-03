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
  libwholegraph
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

# Defined this variable in cpp/CMakeLists.txt
# and used it to gate the `get_nccl.cmake` file
# which is maybe correct?
# but I need to make sure that the nccl I have from the wheel
# is discoverable here
SET(USE_NCCL_WHEEL ON)

rapids_cpm_init()

# include(../../cpp/cmake/thirdparty/get_raft.cmake)

# ##################################################################################################
# * Dependencies  ------------------------------------------------------------

add_subdirectory(../../cpp/ libwholegraph)

if(USE_NCCL_WHEEL)
  set(rpaths "$ORIGIN/../../nvidia/nccl")
  set_property(
    TARGET wholegraph
    PROPERTY INSTALL_RPATH ${rpaths}
    APPEND
  )
endif()

message(STATUS "LIBWHOLEGRAPH: building statically linked wheel")
