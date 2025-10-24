/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <wholememory/tensor_description.h>

namespace graph_ops {
namespace testing {
void gen_node_ids(void* host_target_nodes_ptr,
                  wholememory_array_description_t node_desc,
                  int64_t range,
                  bool unique);
void host_append_unique(void* target_nodes_ptr,
                        wholememory_array_description_t target_nodes_desc,
                        void* neighbor_nodes_ptr,
                        wholememory_array_description_t neighbor_nodes_desc,
                        int* host_total_unique_count,
                        void** host_output_unique_nodes_ptr);

void host_gen_append_unique_neighbor_raw_to_unique(
  void* host_output_unique_nodes_ptr,
  wholememory_array_description_t output_unique_nodes_desc,
  void* host_neighbor_nodes_ptr,
  wholememory_array_description_t neighbor_nodes_desc,
  void** ref_host_output_neighbor_raw_to_unique_mapping_ptr,
  wholememory_array_description_t output_neighbor_raw_to_unique_mapping_desc);

}  // namespace testing
}  // namespace graph_ops
