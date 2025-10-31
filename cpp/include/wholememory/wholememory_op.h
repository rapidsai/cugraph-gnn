/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <wholememory/env_func_ptrs.h>
#include <wholememory/wholememory.h>
#include <wholememory/wholememory_tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Gather Op
 * @param wholememory_tensor : WholeMemory Tensor of embedding table.
 * @param indices_tensor : indices to gather from, should NOT be WholeMemory Tensor
 * @param output_tensor : output tensor to gather to, should NOT be WholeMemoryTensor
 * @param p_env_fns : pointers to environment functions.
 * @param stream : cudaStream_t to use.
 * @param gather_sms : the number of stream multiprocessor used in gather kernel
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_gather(wholememory_tensor_t wholememory_tensor,
                                            wholememory_tensor_t indices_tensor,
                                            wholememory_tensor_t output_tensor,
                                            wholememory_env_func_t* p_env_fns,
                                            void* stream,
                                            int gather_sms = -1);

/**
 * Scatter Op
 * @param input_tensor : input tensor tor scatter from, should NOT be WholeMemory Tensor
 * @param indices_tensor : indices to scatter to, should NOT be WholeMemory Tensor
 * @param wholememory_tensor : WholeMemory Tensor of embedding table.
 * @param p_env_fns : pointers to environment functions.
 * @param stream : cudaStream_t to use.
 * @param scatter_sms : the number of stream multiprocessor used in scatter kernel
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_scatter(wholememory_tensor_t input_tensor,
                                             wholememory_tensor_t indices_tensor,
                                             wholememory_tensor_t wholememory_tensor,
                                             wholememory_env_func_t* p_env_fns,
                                             void* stream,
                                             int scatter_sms = -1);

/**
 * Just a test function,
 * @param input_tensor : input tensor
 * @param output_fixed_tensor : fixed size tensor of output
 * @param output_variable_device_tensor_handle : device version variable tensor
 * @param output_variable_pinned_tensor_handle : pinned version variable tensor
 * @param output_variable_host_tensor_handle : host version variable tensor
 * @param output_variable_entry_count : output entry count
 * @param p_env_fns : pointers to environment functions.
 * @param stream : cudaStream_t to use.
 * @return : wholememory_error_code_t
 */
wholememory_error_code_t wholememory_env_test_op(wholememory_tensor_t input_tensor,
                                                 wholememory_tensor_t output_fixed_tensor,
                                                 void* output_variable_device_tensor_handle,
                                                 void* output_variable_pinned_tensor_handle,
                                                 void* output_variable_host_tensor_handle,
                                                 int64_t output_variable_entry_count,
                                                 wholememory_env_func_t* p_env_fns,
                                                 void* stream);

#ifdef __cplusplus
}
#endif
