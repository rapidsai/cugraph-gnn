/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_runtime_api.h>

#include <wholememory/wholememory.h>

namespace wholememory {

wholememory_error_code_t init(unsigned int flags, LogLevel log_level) noexcept;

wholememory_error_code_t finalize() noexcept;

/**
 * return cudaDeviceProp of dev_id, if dev_id is -1, use current device
 * @param dev_id : device id, -1 for current device
 * @return : cudaDeviceProp pointer
 */
cudaDeviceProp* get_device_prop(int dev_id) noexcept;

}  // namespace wholememory
