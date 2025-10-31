/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholememory/env_func_ptrs.h>
#include <wholememory/tensor_description.h>

namespace wholememory_ops {

class output_memory_handle {
 public:
  explicit output_memory_handle(wholememory_env_func_t* env_fns, void* memory_context)
  {
    output_mem_fns_ = &env_fns->output_fns;
    memory_context_ = memory_context;
  }
  output_memory_handle() = delete;
  ~output_memory_handle() {}

  void* device_malloc(size_t elt_count, wholememory_dtype_t data_type)
  {
    wholememory_tensor_description_t tensor_description;
    get_tensor_description(&tensor_description, elt_count, data_type);
    ptr_ = output_mem_fns_->malloc_fn(
      &tensor_description, WHOLEMEMORY_MA_DEVICE, memory_context_, output_mem_fns_->global_context);
    return ptr_;
  }
  void* device_malloc(wholememory_tensor_description_t* tensor_desc)
  {
    ptr_ = output_mem_fns_->malloc_fn(
      tensor_desc, WHOLEMEMORY_MA_DEVICE, memory_context_, output_mem_fns_->global_context);
    return ptr_;
  }
  void* host_malloc(size_t elt_count, wholememory_dtype_t data_type)
  {
    wholememory_tensor_description_t tensor_description;
    get_tensor_description(&tensor_description, elt_count, data_type);
    ptr_ = output_mem_fns_->malloc_fn(
      &tensor_description, WHOLEMEMORY_MA_HOST, memory_context_, output_mem_fns_->global_context);
    return ptr_;
  }
  void* host_malloc(wholememory_tensor_description_t* tensor_desc)
  {
    ptr_ = output_mem_fns_->malloc_fn(
      tensor_desc, WHOLEMEMORY_MA_HOST, memory_context_, output_mem_fns_->global_context);
    return ptr_;
  }
  void* pinned_malloc(size_t elt_count, wholememory_dtype_t data_type)
  {
    wholememory_tensor_description_t tensor_description;
    get_tensor_description(&tensor_description, elt_count, data_type);
    ptr_ = output_mem_fns_->malloc_fn(
      &tensor_description, WHOLEMEMORY_MA_PINNED, memory_context_, output_mem_fns_->global_context);
    return ptr_;
  }
  void* pinned_malloc(wholememory_tensor_description_t* tensor_desc)
  {
    ptr_ = output_mem_fns_->malloc_fn(
      tensor_desc, WHOLEMEMORY_MA_PINNED, memory_context_, output_mem_fns_->global_context);
    return ptr_;
  }
  void* pointer() const { return ptr_; }

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

  wholememory_output_memory_func_t* output_mem_fns_ = nullptr;
  void* memory_context_                             = nullptr;

  void* ptr_ = nullptr;
};

}  // namespace wholememory_ops
