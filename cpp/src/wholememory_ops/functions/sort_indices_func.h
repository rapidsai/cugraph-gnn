/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>

#include <wholememory_ops/temp_memory_handle.hpp>
#include <wholememory_ops/thrust_allocator.hpp>

namespace wholememory_ops {

wholememory_error_code_t sort_indices_func(const void* indices_before_sort,
                                           wholememory_array_description_t indice_desc,
                                           void* indices_after_sort,
                                           void* raw_indices,
                                           wm_thrust_allocator* p_thrust_allocator,
                                           wholememory_env_func_t* p_env_fns,
                                           cudaStream_t stream);

}  // namespace wholememory_ops
