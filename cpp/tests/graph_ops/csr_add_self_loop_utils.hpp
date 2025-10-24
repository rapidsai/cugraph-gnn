/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <wholememory/tensor_description.h>

namespace graph_ops {
namespace testing {

void gen_local_csr_graph(
  int row_num,
  int col_num,
  int graph_edge_num,
  void* host_csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_desc,
  void* host_csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_desc,
  void* host_csr_weight_ptr                           = nullptr,
  wholememory_array_description_t csr_weight_ptr_desc = wholememory_array_description_t{});

void host_csr_add_self_loop(void* host_csr_row_ptr,
                            wholememory_array_description_t csr_row_ptr_array_desc,
                            void* host_csr_col_ptr,
                            wholememory_array_description_t csr_col_ptr_array_desc,
                            void* host_ref_output_csr_row_ptr,
                            wholememory_array_description_t output_csr_row_ptr_array_desc,
                            void* host_ref_output_csr_col_ptr,
                            wholememory_array_description_t output_csr_col_ptr_array_desc);

}  // namespace testing
}  // namespace graph_ops
