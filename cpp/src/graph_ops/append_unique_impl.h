/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <wholememory/env_func_ptrs.h>
#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

namespace graph_ops {
wholememory_error_code_t graph_append_unique_impl(
  void* target_nodes_ptr,
  wholememory_array_description_t target_nodes_desc,
  void* neighbor_nodes_ptr,
  wholememory_array_description_t neighbor_nodes_desc,
  void* output_unique_node_memory_context,
  int* output_neighbor_raw_to_unique_mapping_ptr,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream);
}
