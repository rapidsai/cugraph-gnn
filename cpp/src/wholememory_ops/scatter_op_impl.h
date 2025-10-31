/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholememory/global_reference.h>
#include <wholememory/wholememory.h>

namespace wholememory_ops {

wholememory_error_code_t wholememory_scatter_mapped(
  void* input,
  wholememory_matrix_description_t input_desc,
  void* indices,
  wholememory_array_description_t indices_desc,
  wholememory_gref_t wholememory_gref,
  wholememory_matrix_description_t wholememory_desc,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream,
  int scatter_sms);

wholememory_error_code_t wholememory_scatter_nccl(void* input,
                                                  wholememory_matrix_description_t input_desc,
                                                  void* indices,
                                                  wholememory_array_description_t indices_desc,
                                                  wholememory_handle_t wholememory_handle,
                                                  wholememory_matrix_description_t wholememory_desc,
                                                  wholememory_env_func_t* p_env_fns,
                                                  cudaStream_t stream,
                                                  int scatter_sms);

wholememory_error_code_t wholememory_scatter_distributed(
  void* input,
  wholememory_matrix_description_t input_desc,
  void* indices,
  wholememory_array_description_t indices_desc,
  wholememory_handle_t wholememory_handle,
  wholememory_matrix_description_t wholememory_desc,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream,
  int scatter_sms);

#ifdef WITH_NVSHMEM_SUPPORT
wholememory_error_code_t wholememory_scatter_nvshmem(
  void* input,
  wholememory_matrix_description_t input_desc,
  void* indices,
  wholememory_array_description_t indices_desc,
  wholememory_handle_t wholememory_handle,
  wholememory_matrix_description_t wholememory_desc,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream,
  int scatter_sms);

#endif
}  // namespace wholememory_ops
