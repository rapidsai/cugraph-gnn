/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda_runtime_api.h>

#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>

#include "cuda_macros.hpp"
#include "wholememory_ops/functions/gather_scatter_func.h"
#include "wholememory_ops/functions/sort_indices_func.h"
#include "wholememory_ops/temp_memory_handle.hpp"
#include "wholememory_ops/thrust_allocator.hpp"

namespace wholememory_ops {

wholememory_error_code_t wholememory_gather_mapped(
  wholememory_gref_t wholememory_gref,
  wholememory_matrix_description_t wholememory_desc,
  void* indices,
  wholememory_array_description_t indice_desc,
  void* output,
  wholememory_matrix_description_t output_desc,
  bool gather_with_sorted_ids,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream,
  int gather_sms)
{
  if (gather_with_sorted_ids) {
    wm_thrust_allocator thrust_allocator(p_env_fns);
    temp_memory_handle dev_indices_after_sort(p_env_fns);
    void* dev_indices_after_sort_ptr =
      dev_indices_after_sort.device_malloc(indice_desc.size, indice_desc.dtype);
    temp_memory_handle dev_raw_indices(p_env_fns);
    void* dev_raw_indices_ptr = dev_raw_indices.device_malloc(indice_desc.size, indice_desc.dtype);
    auto raw_indices_desc = wholememory_create_array_desc(indice_desc.size, 0, indice_desc.dtype);
    WHOLEMEMORY_RETURN_ON_FAIL(sort_indices_func(indices,
                                                 indice_desc,
                                                 dev_indices_after_sort_ptr,
                                                 dev_raw_indices_ptr,
                                                 &thrust_allocator,
                                                 p_env_fns,
                                                 stream));
    WHOLEMEMORY_RETURN_ON_FAIL(gather_with_sorted_ids_func(wholememory_gref,
                                                           wholememory_desc,
                                                           dev_indices_after_sort_ptr,
                                                           indice_desc,
                                                           dev_raw_indices_ptr,
                                                           raw_indices_desc,
                                                           output,
                                                           output_desc,
                                                           stream,
                                                           gather_sms));
  } else {
    WHOLEMEMORY_RETURN_ON_FAIL(gather_func(wholememory_gref,
                                           wholememory_desc,
                                           indices,
                                           indice_desc,
                                           output,
                                           output_desc,
                                           stream,
                                           gather_sms));
  }
  WM_CUDA_DEBUG_SYNC_STREAM(stream);
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops
