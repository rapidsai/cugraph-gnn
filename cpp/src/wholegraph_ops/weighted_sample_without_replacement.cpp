/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <wholememory/wholegraph_op.h>

#include <wholegraph_ops/weighted_sample_without_replacement_impl.h>

#include "error.hpp"
#include "logger.hpp"

wholememory_error_code_t wholegraph_csr_weighted_sample_without_replacement(
  wholememory_tensor_t wm_csr_row_ptr_tensor,
  wholememory_tensor_t wm_csr_col_ptr_tensor,
  wholememory_tensor_t wm_csr_weight_ptr_tensor,
  wholememory_tensor_t center_nodes_tensor,
  int max_sample_count,
  wholememory_tensor_t output_sample_offset_tensor,
  void* output_dest_memory_context,
  void* output_center_localid_memory_context,
  void* output_edge_gid_memory_context,
  unsigned long long random_seed,
  wholememory_env_func_t* p_env_fns,
  void* stream)
{
  bool const csr_row_ptr_has_handle = wholememory_tensor_has_handle(wm_csr_row_ptr_tensor);
  wholememory_memory_type_t csr_row_ptr_memory_type = WHOLEMEMORY_MT_NONE;
  if (csr_row_ptr_has_handle) {
    csr_row_ptr_memory_type =
      wholememory_get_memory_type(wholememory_tensor_get_memory_handle(wm_csr_row_ptr_tensor));
  }
  WHOLEMEMORY_EXPECTS_NOTHROW(!csr_row_ptr_has_handle ||
                                csr_row_ptr_memory_type == WHOLEMEMORY_MT_CHUNKED ||
                                csr_row_ptr_memory_type == WHOLEMEMORY_MT_CONTINUOUS,
                              "Memory type not supported.");
  bool const csr_col_ptr_has_handle = wholememory_tensor_has_handle(wm_csr_col_ptr_tensor);
  wholememory_memory_type_t csr_col_ptr_memory_type = WHOLEMEMORY_MT_NONE;
  if (csr_col_ptr_has_handle) {
    csr_col_ptr_memory_type =
      wholememory_get_memory_type(wholememory_tensor_get_memory_handle(wm_csr_col_ptr_tensor));
  }
  WHOLEMEMORY_EXPECTS_NOTHROW(!csr_col_ptr_has_handle ||
                                csr_col_ptr_memory_type == WHOLEMEMORY_MT_CHUNKED ||
                                csr_col_ptr_memory_type == WHOLEMEMORY_MT_CONTINUOUS,
                              "Memory type not supported.");
  bool const csr_weight_ptr_has_handle = wholememory_tensor_has_handle(wm_csr_weight_ptr_tensor);
  wholememory_memory_type_t csr_weight_ptr_memory_type = WHOLEMEMORY_MT_NONE;
  if (csr_weight_ptr_has_handle) {
    csr_weight_ptr_memory_type =
      wholememory_get_memory_type(wholememory_tensor_get_memory_handle(wm_csr_weight_ptr_tensor));
  }
  WHOLEMEMORY_EXPECTS_NOTHROW(!csr_weight_ptr_has_handle ||
                                csr_weight_ptr_memory_type == WHOLEMEMORY_MT_CHUNKED ||
                                csr_weight_ptr_memory_type == WHOLEMEMORY_MT_CONTINUOUS,
                              "Memory type not supported.");

  auto csr_row_ptr_tensor_description =
    *wholememory_tensor_get_tensor_description(wm_csr_row_ptr_tensor);
  auto csr_col_ptr_tensor_description =
    *wholememory_tensor_get_tensor_description(wm_csr_col_ptr_tensor);
  auto csr_weight_ptr_tensor_description =
    *wholememory_tensor_get_tensor_description(wm_csr_weight_ptr_tensor);
  if (csr_row_ptr_tensor_description.dim != 1) {
    WHOLEMEMORY_ERROR("wm_csr_row_ptr_tensor should be 1D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (csr_col_ptr_tensor_description.dim != 1) {
    WHOLEMEMORY_ERROR("wm_csr_col_ptr_tensor should be 1D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  if (csr_weight_ptr_tensor_description.dim != 1) {
    WHOLEMEMORY_ERROR("wm_csr_weight_ptr_tensor should be 1D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  wholememory_array_description_t wm_csr_row_ptr_desc, wm_csr_col_ptr_desc, wm_csr_weight_ptr_desc;
  if (!wholememory_convert_tensor_desc_to_array(&wm_csr_row_ptr_desc,
                                                &csr_row_ptr_tensor_description)) {
    WHOLEMEMORY_ERROR("Input wm_csr_row_ptr_tensor convert to array failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (!wholememory_convert_tensor_desc_to_array(&wm_csr_col_ptr_desc,
                                                &csr_col_ptr_tensor_description)) {
    WHOLEMEMORY_ERROR("Input wm_csr_col_ptr_tensor convert to array failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }
  if (!wholememory_convert_tensor_desc_to_array(&wm_csr_weight_ptr_desc,
                                                &csr_weight_ptr_tensor_description)) {
    WHOLEMEMORY_ERROR("Input wm_csr_weight_ptr_tensor convert to array failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  wholememory_tensor_description_t center_nodes_tensor_desc =
    *wholememory_tensor_get_tensor_description(center_nodes_tensor);
  if (center_nodes_tensor_desc.dim != 1) {
    WHOLEMEMORY_ERROR("Input center_nodes_tensor should be 1D tensor");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  wholememory_array_description_t center_nodes_desc;
  if (!wholememory_convert_tensor_desc_to_array(&center_nodes_desc, &center_nodes_tensor_desc)) {
    WHOLEMEMORY_ERROR("Input center_nodes_tensor convert to array failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  wholememory_tensor_description_t output_sample_offset_tensor_desc =
    *wholememory_tensor_get_tensor_description(output_sample_offset_tensor);
  if (output_sample_offset_tensor_desc.dim != 1) {
    WHOLEMEMORY_ERROR("Output output_sample_offset_tensor should be 1D tensor.");
    return WHOLEMEMORY_INVALID_INPUT;
  }
  wholememory_array_description_t output_sample_offset_desc;
  if (!wholememory_convert_tensor_desc_to_array(&output_sample_offset_desc,
                                                &output_sample_offset_tensor_desc)) {
    WHOLEMEMORY_ERROR("Output output_sample_offset_tensor convert to array failed.");
    return WHOLEMEMORY_LOGIC_ERROR;
  }

  void* center_nodes         = wholememory_tensor_get_data_pointer(center_nodes_tensor);
  void* output_sample_offset = wholememory_tensor_get_data_pointer(output_sample_offset_tensor);
  wholememory_gref_t wm_csr_row_ptr_gref, wm_csr_col_ptr_gref, wm_csr_weight_ptr_gref;
  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_tensor_get_global_reference(wm_csr_row_ptr_tensor, &wm_csr_row_ptr_gref));
  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_tensor_get_global_reference(wm_csr_col_ptr_tensor, &wm_csr_col_ptr_gref));
  WHOLEMEMORY_RETURN_ON_FAIL(
    wholememory_tensor_get_global_reference(wm_csr_weight_ptr_tensor, &wm_csr_weight_ptr_gref));

  return wholegraph_ops::wholegraph_csr_weighted_sample_without_replacement_mapped(
    wm_csr_row_ptr_gref,
    wm_csr_row_ptr_desc,
    wm_csr_col_ptr_gref,
    wm_csr_col_ptr_desc,
    wm_csr_weight_ptr_gref,
    wm_csr_weight_ptr_desc,
    center_nodes,
    center_nodes_desc,
    max_sample_count,
    output_sample_offset,
    output_sample_offset_desc,
    output_dest_memory_context,
    output_center_localid_memory_context,
    output_edge_gid_memory_context,
    random_seed,
    p_env_fns,
    static_cast<cudaStream_t>(stream));
}
