/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "csr_add_self_loop_func.cuh"
#include <wholememory/wholememory.h>

#include "wholememory_ops/register.hpp"

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
  cudaStream_t stream)
{
  try {
    csr_add_self_loop_func(csr_row_ptr,
                           csr_row_ptr_array_desc,
                           csr_col_ptr,
                           csr_col_ptr_array_desc,
                           output_csr_row_ptr,
                           output_csr_row_ptr_array_desc,
                           output_csr_col_ptr,
                           output_csr_col_ptr_array_desc,
                           stream);

  } catch (const wholememory::cuda_error& rle) {
    // WHOLEMEMORY_FAIL_NOTHROW("%s", rle.what());
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (const wholememory::logic_error& le) {
    return WHOLEMEMORY_LOGIC_ERROR;
  } catch (...) {
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace graph_ops
