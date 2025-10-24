/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "sort_unique_ids_for_hierarchy_func.h"
#include "sort_unique_indices_func.h"

#include <cassert>
#include <cstdint>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <thrust/sequence.h>
#include <thrust/unique.h>

#include <wholememory/wholememory.h>

#include "cuda_macros.hpp"
#include "error.hpp"
#include "logger.hpp"
#include "wholememory/communicator.hpp"
#include "wholememory/integer_utils.hpp"
#include "wholememory_ops/register.hpp"
#include "wholememory_ops/temp_memory_handle.hpp"
#include <wholememory_ops/thrust_allocator.hpp>

namespace wholememory_ops {

template <typename IndexT>
__global__ void SortUniqueIndiceMapKernel(IndexT* indice_map,
                                          size_t indice_count,
                                          const IndexT* sort_raw_indices,
                                          const int* unique_count_ptr,
                                          const IndexT* unique_offset_ptr,
                                          size_t num_unique)
{
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < indice_count;
       idx += blockDim.x * gridDim.x) {
    if (idx >= num_unique) break;
    IndexT offset = unique_offset_ptr[idx];
    int count     = unique_count_ptr[idx];
    for (IndexT i = offset; i < offset + count; i++) {
      indice_map[sort_raw_indices[i]] = idx;
    }
  }
}

template <typename IndexT>
void SortUniqueIndicesMapTempFunc(void* indice_map,
                                  wholememory_array_description_t indice_desc,
                                  const void* sort_raw_indices,
                                  const int* unique_count_ptr,
                                  size_t num_unique,
                                  wm_thrust_allocator* p_thrust_allocator,
                                  wholememory_env_func_t* p_env_fns,
                                  cudaStream_t stream)
{
  static constexpr int BLOCK_SIZE = 128;
  int block_count                 = wholememory::div_rounding_up_unsafe(num_unique, BLOCK_SIZE);

  temp_memory_handle dev_unique_offset_handle(p_env_fns);
  IndexT* unique_offset_ptr =
    static_cast<IndexT*>(dev_unique_offset_handle.device_malloc(num_unique, indice_desc.dtype));
  IndexT* indice_map_ptr             = static_cast<IndexT*>(indice_map);
  const IndexT* sort_raw_indices_ptr = static_cast<const IndexT*>(sort_raw_indices);

  void* cub_temp_storage    = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
    cub_temp_storage, temp_storage_bytes, unique_count_ptr, unique_offset_ptr, num_unique, stream);
  cub_temp_storage = p_thrust_allocator->allocate(temp_storage_bytes);
  cub::DeviceScan::ExclusiveSum(
    cub_temp_storage, temp_storage_bytes, unique_count_ptr, unique_offset_ptr, num_unique, stream);
  SortUniqueIndiceMapKernel<<<block_count, BLOCK_SIZE, 0, stream>>>(indice_map_ptr,
                                                                    indice_desc.size,
                                                                    sort_raw_indices_ptr,
                                                                    unique_count_ptr,
                                                                    unique_offset_ptr,
                                                                    num_unique);
  p_thrust_allocator->deallocate(reinterpret_cast<char*>(cub_temp_storage), temp_storage_bytes);
}

REGISTER_DISPATCH_ONE_TYPE(SortUniqueIndicesMapTempFunc, SortUniqueIndicesMapTempFunc, SINT3264)

wholememory_error_code_t sort_unique_ids_for_hierarchy_func(
  void* indices,
  wholememory_array_description_t indice_desc,
  temp_memory_handle* output_indices_handle,
  wholememory_array_description_t* output_indices_desc,
  temp_memory_handle* dev_indice_map_handle,
  wm_thrust_allocator* p_thrust_allocator,
  wholememory_env_func_t* p_env_fns,
  cudaStream_t stream)
{
  if (indice_desc.size == 0) {
    *output_indices_desc = wholememory_create_array_desc(0, 0, indice_desc.dtype);
    return WHOLEMEMORY_SUCCESS;
  }
  int num_runs = 0;
  temp_memory_handle unique_count_handle(p_env_fns);
  temp_memory_handle dev_sort_raw_indices_handle(p_env_fns);
  void* dev_sort_raw_indices_ptr =
    dev_sort_raw_indices_handle.device_malloc(indice_desc.size, indice_desc.dtype);
  sort_unique_indices_func(indices,
                           indice_desc,
                           dev_sort_raw_indices_ptr,
                           &num_runs,
                           output_indices_handle,
                           &unique_count_handle,
                           p_thrust_allocator,
                           p_env_fns,
                           stream);
  *output_indices_desc = wholememory_create_array_desc(num_runs, 0, indice_desc.dtype);
  void* dev_indice_map_ptr =
    dev_indice_map_handle->device_malloc(indice_desc.size, indice_desc.dtype);
  WM_CUDA_CHECK(cudaGetLastError());
  try {
    DISPATCH_ONE_TYPE(indice_desc.dtype,
                      SortUniqueIndicesMapTempFunc,
                      dev_indice_map_ptr,
                      indice_desc,
                      dev_sort_raw_indices_ptr,
                      static_cast<int*>(unique_count_handle.pointer()),
                      num_runs,
                      p_thrust_allocator,
                      p_env_fns,
                      stream);
  } catch (...) {
    WHOLEMEMORY_FAIL_NOTHROW("map indices failed");
  }
  return WHOLEMEMORY_SUCCESS;
}

}  // namespace wholememory_ops
