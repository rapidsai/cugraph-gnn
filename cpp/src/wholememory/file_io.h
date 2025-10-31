/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholememory/wholememory.h>

namespace wholememory {

wholememory_error_code_t load_file_to_handle(wholememory_handle_t wholememory_handle,
                                             size_t memory_offset,
                                             size_t memory_entry_stride,
                                             size_t entry_size,
                                             const char** file_names,
                                             int file_count,
                                             int round_robin_size) noexcept;

wholememory_error_code_t store_handle_to_file(wholememory_handle_t wholememory_handle,
                                              size_t memory_offset,
                                              size_t memory_entry_stride,
                                              size_t entry_size,
                                              const char* local_file_name) noexcept;

}  // namespace wholememory
