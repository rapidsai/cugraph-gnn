/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_runtime_api.h>
#include <wholememory/tensor_description.h>

/**
 * Function pointers for memory allocation.
 * Input tensor memory should be allocated and use void* pointer to the memory and
 * wholememory_array_description_t or wholememory_matrix_description_t to specify the shape Output
 * tensor with fixed size should be the same as Input tensor. Output tensor with shape determined by
 * Op should has void* memory_context input and allocated by wholememory_malloc_func_t functions.
 */

#ifdef __cplusplus
extern "C" {
#endif

enum wholememory_memory_allocation_type_t {
  WHOLEMEMORY_MA_NONE = 0,
  WHOLEMEMORY_MA_DEVICE,
  WHOLEMEMORY_MA_HOST,
  WHOLEMEMORY_MA_PINNED,
};

/**
 * Function pointer to create temporary memory context.
 */
typedef void (*wholememory_create_memory_context_func_t)(void** memory_context,
                                                         void* global_context);

typedef void (*wholememory_destroy_memory_context_func_t)(void* memory_context,
                                                          void* global_context);

typedef void* (*wholememory_malloc_func_t)(
  wholememory_tensor_description_t* desc,
  wholememory_memory_allocation_type_t memory_allocation_type,
  void* memory_context,
  void* global_context);

typedef void (*wholememory_free_func_t)(void* memory_context, void* global_context);

struct wholememory_temp_memory_func_t {
  wholememory_create_memory_context_func_t create_memory_context_fn;
  wholememory_destroy_memory_context_func_t destroy_memory_context_fn;
  wholememory_malloc_func_t malloc_fn;
  wholememory_free_func_t free_fn;
  void* global_context;
};
struct wholememory_output_memory_func_t {
  wholememory_malloc_func_t malloc_fn;
  wholememory_free_func_t free_fn;
  void* global_context;
};

struct wholememory_env_func_t {
  wholememory_temp_memory_func_t temporary_fns; /* function pointers to create temporary memory */
  wholememory_output_memory_func_t output_fns;  /* function pointers to create Op output memory */
};

cudaDeviceProp* get_device_prop(int dev_id);

#ifdef __cplusplus
}
#endif
