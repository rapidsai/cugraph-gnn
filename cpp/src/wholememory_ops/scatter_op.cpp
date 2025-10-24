/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <wholememory/wholememory_op.h>

#include <wholememory_ops/scatter_op_impl.h>

#include "error.hpp"
#include "logger.hpp"

wholememory_error_code_t wholememory_scatter(wholememory_tensor_t input_tensor,
                                             wholememory_tensor_t indices_tensor,
                                             wholememory_tensor_t wholememory_tensor,
                                             wholememory_env_func_t* p_env_fns,
                                             void* stream,
                                             int scatter_sms)
{
  bool const has_handle                 = wholememory_tensor_has_handle(wholememory_tensor);
  wholememory_memory_type_t memory_type = WHOLEMEMORY_MT_NONE;
  if (has_handle) {
    memory_type =
      wholememory_get_memory_type(wholememory_tensor_get_memory_handle(wholememory_tensor));
  }
  wholememory_matrix_description_t matrix_description;
  auto tensor_description = *wholememory_tensor_get_tensor_description(wholememory_tensor);
  if (tensor_description.dim != 1 && tensor_description.dim != 2) {
    WHOLEMEMORY_ERROR("wholememory_tensor should be 1D or 2D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (tensor_description.dim == 1) {
    if (!wholememory_unsqueeze_tensor(&tensor_description, 1)) {
      WHOLEMEMORY_ERROR("Output 1D wholememory_tensor unsqueeze to 2D failed.");
      return WHOLEMEMORY_INVALID_INPUT;
    }
  }
  if (!wholememory_convert_tensor_desc_to_matrix(&matrix_description, &tensor_description)) {
    WHOLEMEMORY_ERROR("Output wholememory_tensor convert to matrix failed.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (wholememory_tensor_get_tensor_description(indices_tensor)->dim != 1) {
    WHOLEMEMORY_ERROR("indices tensor should be 1D tensor");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  wholememory_tensor_description_t input_tensor_desc =
    *wholememory_tensor_get_tensor_description(input_tensor);
  if (input_tensor_desc.dim != tensor_description.dim) {
    WHOLEMEMORY_ERROR("input tensor should be same dim as wholememory_tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (input_tensor_desc.dim == 1) {
    if (!wholememory_unsqueeze_tensor(&input_tensor_desc, 1)) {
      WHOLEMEMORY_ERROR("Input 1D wholememory_tensor unsqueeze to 2D failed.");
      return WHOLEMEMORY_LOGIC_ERROR;
    }
  }
  void* indices = wholememory_tensor_get_data_pointer(indices_tensor);
  void* input   = wholememory_tensor_get_data_pointer(input_tensor);
  wholememory_array_description_t indices_desc;
  wholememory_matrix_description_t input_desc;
  if (!wholememory_convert_tensor_desc_to_array(
        &indices_desc, wholememory_tensor_get_tensor_description(indices_tensor))) {
    WHOLEMEMORY_ERROR("Convert indices tensor to array failed.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (!wholememory_convert_tensor_desc_to_matrix(&input_desc, &input_tensor_desc)) {
    WHOLEMEMORY_ERROR("Convert input tensor to matrix failed.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (has_handle && memory_type == WHOLEMEMORY_MT_DISTRIBUTED) {
    return wholememory_ops::wholememory_scatter_distributed(
      input,
      input_desc,
      indices,
      indices_desc,
      wholememory_tensor_get_memory_handle(wholememory_tensor),
      matrix_description,
      p_env_fns,
      static_cast<cudaStream_t>(stream),
      scatter_sms);
  }

  WHOLEMEMORY_EXPECTS_NOTHROW(!has_handle || memory_type == WHOLEMEMORY_MT_CHUNKED ||
                                memory_type == WHOLEMEMORY_MT_CONTINUOUS,
                              "Memory type not supported.");

  wholememory_gref_t gref;
  WHOLEMEMORY_RETURN_ON_FAIL(wholememory_tensor_get_global_reference(wholememory_tensor, &gref));

  return wholememory_ops::wholememory_scatter_mapped(input,
                                                     input_desc,
                                                     indices,
                                                     indices_desc,
                                                     gref,
                                                     matrix_description,
                                                     p_env_fns,
                                                     static_cast<cudaStream_t>(stream),
                                                     scatter_sms);
}
