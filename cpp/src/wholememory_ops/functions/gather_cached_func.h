/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

namespace wholememory_ops {

wholememory_error_code_t gather_cached_func(wholememory_gref_t padded_embedding_gref,
                                            wholememory_tensor_description_t* embedding_desc,
                                            wholememory_gref_t cached_embedding_gref,
                                            wholememory_tensor_description_t* cached_embedding_desc,
                                            wholememory_gref_t cache_line_tag_gref,
                                            void* indices,
                                            wholememory_tensor_description_t* indices_desc,
                                            void* output,
                                            wholememory_tensor_description_t* output_desc,
                                            int cache_set_coverage,
                                            int64_t cache_start_gid,
                                            int64_t raw_start_gid,
                                            cudaStream_t stream);

wholememory_error_code_t try_gather_cached_func(
  wholememory_gref_t cached_embedding_gref,
  wholememory_tensor_description_t* cached_embedding_desc,
  wholememory_gref_t cache_line_tag_gref,
  void* indices,
  wholememory_tensor_description_t* indices_desc,
  void* hit_indices,
  void* miss_indices,
  void* output,
  wholememory_tensor_description_t* output_desc,
  int cache_set_coverage,
  int64_t cache_start_gid,
  cudaStream_t stream);

}  // namespace wholememory_ops
