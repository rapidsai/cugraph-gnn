// clang-format off
// SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format on
#pragma once
#include <cuda.h>

#if CUDA_VERSION >= 12030
#include <nvml.h>

bool NvmlFabricSymbolLoaded();

typedef nvmlReturn_t (*nvmlDeviceGetHandleByIndexFunc)(unsigned int, nvmlDevice_t*);
typedef nvmlReturn_t (*nvmlDeviceGetGpuFabricInfoFunc)(nvmlDevice_t, nvmlGpuFabricInfo_t*);

extern nvmlDeviceGetHandleByIndexFunc nvmlDeviceGetHandleByIndexPtr;
extern nvmlDeviceGetGpuFabricInfoFunc nvmlDeviceGetGpuFabricInfoPtr;
#endif  // CUDA_VERSION >= 12030
