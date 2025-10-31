/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>
#include <wholememory_ops/thrust_allocator.hpp>

#include "wholememory_ops/temp_memory_handle.hpp"

namespace wholememory_ops {

wholememory_error_code_t bucket_and_reorder_ids_for_hierarchy_func(
  void* indices,
  wholememory_array_description_t indice_desc,
  void* dev_bucket_indices,
  void* dev_indice_map,
  int64_t* host_bucket_id_count,
  size_t* dev_embedding_entry_offsets,
  wholememory_comm_t wm_global_comm,
  wholememory_comm_t wm_local_comm,
  int bucket_cross_or_local,  // 0: cross, 1: local
  wm_thrust_allocator* p_thrust_allocator,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream);

wholememory_error_code_t bucket_local_ids_func(void* indices,
                                               wholememory_array_description_t indice_desc,
                                               int64_t* host_bucket_id_count,
                                               size_t* dev_embedding_entry_offsets,
                                               wholememory_comm_t wm_local_comm,
                                               wholememory_comm_t wm_cross_comm,
                                               wm_thrust_allocator* p_thrust_allocator,
                                               wholememory_env_func_t* p_env_fns,
                                               cudaStream_t stream);

}  // namespace wholememory_ops
