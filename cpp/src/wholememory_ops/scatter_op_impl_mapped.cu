/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda_runtime_api.h>

#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>

#include "cuda_macros.hpp"
#include "wholememory_ops/functions/gather_scatter_func.h"

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
  int scatter_sms)
{
  return scatter_func(input,
                      input_desc,
                      indices,
                      indices_desc,
                      wholememory_gref,
                      wholememory_desc,
                      stream,
                      scatter_sms);
  WM_CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace wholememory_ops
