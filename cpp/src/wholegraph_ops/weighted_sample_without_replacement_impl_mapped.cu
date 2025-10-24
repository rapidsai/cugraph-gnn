/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda_runtime_api.h>

#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>

#include "weighted_sample_without_replacement_func.cuh"
#include "wholememory_ops/register.hpp"

namespace wholegraph_ops {

REGISTER_DISPATCH_THREE_TYPES(WeightedSampleWithoutReplacementCSR,
                              wholegraph_csr_weighted_sample_without_replacement_func,
                              SINT3264,
                              SINT3264,
                              FLOAT_DOUBLE)

wholememory_error_code_t wholegraph_csr_weighted_sample_without_replacement_mapped(
  wholememory_gref_t wm_csr_row_ptr,
  wholememory_array_description_t wm_csr_row_ptr_desc,
  wholememory_gref_t wm_csr_col_ptr,
  wholememory_array_description_t wm_csr_col_ptr_desc,
  wholememory_gref_t wm_csr_weight_ptr,
  wholememory_array_description_t wm_csr_weight_ptr_desc,
  void* center_nodes,
  wholememory_array_description_t center_nodes_desc,
  int max_sample_count,
  void* output_sample_offset,
  wholememory_array_description_t output_sample_offset_desc,
  void* output_dest_memory_context,
  void* output_center_localid_memory_context,
  void* output_edge_gid_memory_context,
  unsigned long long random_seed,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream)
{
  try {
    DISPATCH_THREE_TYPES(center_nodes_desc.dtype,
                         wm_csr_col_ptr_desc.dtype,
                         wm_csr_weight_ptr_desc.dtype,
                         WeightedSampleWithoutReplacementCSR,
                         wm_csr_row_ptr,
                         wm_csr_row_ptr_desc,
                         wm_csr_col_ptr,
                         wm_csr_col_ptr_desc,
                         wm_csr_weight_ptr,
                         wm_csr_weight_ptr_desc,
                         center_nodes,
                         center_nodes_desc,
                         max_sample_count,
                         output_sample_offset,
                         output_sample_offset_desc,
                         output_dest_memory_context,
                         output_center_localid_memory_context,
                         output_edge_gid_memory_context,
                         random_seed,
                         p_env_fns,
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

}  // namespace wholegraph_ops
