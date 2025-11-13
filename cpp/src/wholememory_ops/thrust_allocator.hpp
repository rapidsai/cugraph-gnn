/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <map>

#include <wholememory/env_func_ptrs.h>

namespace wholememory_ops {

class wm_thrust_allocator {
 public:
  using value_type = char;
  explicit wm_thrust_allocator(wholememory_env_func_t* fns) : fns(fns) {}
  wm_thrust_allocator() = delete;
  ~wm_thrust_allocator();

  value_type* allocate(std::ptrdiff_t mem_size);
  void deallocate(value_type* p, size_t mem_size);
  void deallocate_all();

  wholememory_env_func_t* fns;
  std::map<value_type*, void*> mem_ptr_to_context_map;
};

}  // namespace wholememory_ops
