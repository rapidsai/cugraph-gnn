/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "wholememory_ops/temp_memory_handle.hpp"
#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>
#include <wholememory_ops/thrust_allocator.hpp>

namespace wholememory_ops {

wholememory_error_code_t sort_unique_ids_for_hierarchy_func(
  void* indices,
  wholememory_array_description_t indice_desc,
  temp_memory_handle* output_indices_handle,
  wholememory_array_description_t* output_indices_desc,
  temp_memory_handle* dev_indice_map_handle,  // indice_desc
  wm_thrust_allocator* p_thrust_allocator,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream);

}  // namespace wholememory_ops
