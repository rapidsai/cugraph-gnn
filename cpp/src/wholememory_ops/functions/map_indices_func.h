/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <wholememory/tensor_description.h>
#include <wholememory/wholememory.h>
#include <wholememory/wholememory_tensor.h>

namespace wholememory_ops {

wholememory_error_code_t storage_index2wm_embedding_index(wholememory_tensor_t indices,
                                                          wholememory_tensor_t mapped_indices,
                                                          wholememory_tensor_t allocated_embedding,
                                                          int round_robin_size,
                                                          int64_t stream_int);

}  // namespace wholememory_ops
