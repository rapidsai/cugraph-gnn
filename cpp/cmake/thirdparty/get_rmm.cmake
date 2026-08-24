# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# This function finds RMM and sets any additional necessary environment variables.
function(find_and_configure_rmm)
  include(${rapids-cmake-dir}/cpm/rmm.cmake)

  rapids_cpm_rmm(BUILD_EXPORT_SET wholegraph-exports INSTALL_EXPORT_SET wholegraph-exports)

  # Propagate RMM source/binary dirs to parent scope.
  set(rmm_SOURCE_DIR
      "${rmm_SOURCE_DIR}"
      PARENT_SCOPE
  )
  set(rmm_BINARY_DIR
      "${rmm_BINARY_DIR}"
      PARENT_SCOPE
  )
endfunction()

find_and_configure_rmm()
