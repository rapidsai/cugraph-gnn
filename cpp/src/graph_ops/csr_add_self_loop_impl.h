/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cuda_runtime_api.h>
#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

namespace graph_ops {
wholememory_error_code_t csr_add_self_loop_impl(
  int* csr_row_ptr,
  wholememory_array_description_t csr_row_ptr_array_desc,
  int* csr_col_ptr,
  wholememory_array_description_t csr_col_ptr_array_desc,
  int* output_csr_row_ptr,
  wholememory_array_description_t output_csr_row_ptr_array_desc,
  int* output_csr_col_ptr,
  wholememory_array_description_t output_csr_col_ptr_array_desc,
  cudaStream_t stream);

}  // namespace graph_ops
