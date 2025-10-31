/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholememory/env_func_ptrs.h>
#include <wholememory/tensor_description.h>

namespace wholememory_ops {

class temp_memory_handle {
 public:
  explicit temp_memory_handle(wholememory_env_func_t* env_fns)
  {
    temp_mem_fns_ = &env_fns->temporary_fns;
    temp_mem_fns_->create_memory_context_fn(&memory_context_, temp_mem_fns_->global_context);
  }
  temp_memory_handle() = delete;
  ~temp_memory_handle() { free_memory(); }
  void* device_malloc(size_t elt_count, wholememory_dtype_t data_type)
  {
    free_data();
    wholememory_tensor_description_t tensor_description;
    get_tensor_description(&tensor_description, elt_count, data_type);
    ptr_ = temp_mem_fns_->malloc_fn(
      &tensor_description, WHOLEMEMORY_MA_DEVICE, memory_context_, temp_mem_fns_->global_context);
    return ptr_;
  }
  void* host_malloc(size_t elt_count, wholememory_dtype_t data_type)
  {
    free_data();
    wholememory_tensor_description_t tensor_description;
    get_tensor_description(&tensor_description, elt_count, data_type);
    ptr_ = temp_mem_fns_->malloc_fn(
      &tensor_description, WHOLEMEMORY_MA_HOST, memory_context_, temp_mem_fns_->global_context);
    return ptr_;
  }
  void* pinned_malloc(size_t elt_count, wholememory_dtype_t data_type)
  {
    free_data();
    wholememory_tensor_description_t tensor_description;
    get_tensor_description(&tensor_description, elt_count, data_type);
    ptr_ = temp_mem_fns_->malloc_fn(
      &tensor_description, WHOLEMEMORY_MA_PINNED, memory_context_, temp_mem_fns_->global_context);
    return ptr_;
  }
  [[nodiscard]] void* pointer() const { return ptr_; }
  void free_data()
  {
    if (ptr_ != nullptr) {
      temp_mem_fns_->free_fn(memory_context_, temp_mem_fns_->global_context);
      ptr_ = nullptr;
    }
  }
  void free_memory()
  {
    free_data();
    if (memory_context_ != nullptr) {
      temp_mem_fns_->destroy_memory_context_fn(memory_context_, temp_mem_fns_->global_context);
      memory_context_ = nullptr;
    }
  }

 private:
  static void get_tensor_description(wholememory_tensor_description_t* tensor_description,
                                     size_t elt_count,
                                     wholememory_dtype_t data_type)
  {
    wholememory_initialize_tensor_desc(tensor_description);
    tensor_description->dim            = 1;
    tensor_description->storage_offset = 0;
    tensor_description->dtype          = data_type;
    tensor_description->sizes[0]       = elt_count;
    tensor_description->strides[0]     = 1;
  }

  wholememory_temp_memory_func_t* temp_mem_fns_ = nullptr;
  void* memory_context_                         = nullptr;

  void* ptr_ = nullptr;
};

}  // namespace wholememory_ops
