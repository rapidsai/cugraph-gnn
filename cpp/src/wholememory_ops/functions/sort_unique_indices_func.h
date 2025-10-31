/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

#include <wholememory_ops/temp_memory_handle.hpp>
#include <wholememory_ops/thrust_allocator.hpp>

namespace wholememory_ops {

wholememory_error_code_t sort_unique_indices_func(const void* indices,
                                                  wholememory_array_description_t indice_desc,
                                                  void* sort_raw_indices,
                                                  int* num_runs,
                                                  temp_memory_handle* unique_indices_handle,
                                                  temp_memory_handle* unique_count_handle,
                                                  wm_thrust_allocator* p_thrust_allocator,
                                                  wholememory_env_func_t* p_env_fns,
                                                  cudaStream_t stream);

}  // namespace wholememory_ops
