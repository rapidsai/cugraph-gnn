/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "wholememory/wholememory.h"

#if CUDA_VERSION >= 12030
#include "nvml_wrap.h"
#include <nvml.h>
#endif
bool DevAttrPagebleMemoryAccess();

bool DevAttrCanUseHostPointerForRegisteredMem();

bool DeviceCanAccessPeer(int peer_device);

bool DevicesCanAccessP2P(const int* dev_ids, int count);

int GetCudaCompCap();

const char* GetCPUArch();

bool SupportMNNVL();

bool SupportEGM();

// bool SupportMNNVLForEGM();
#if CUDA_VERSION >= 12030
namespace wholememory {

inline bool nvmlFabricSymbolLoaded = NvmlFabricSymbolLoaded();
wholememory_error_code_t GetGpuFabricInfo(int dev, nvmlGpuFabricInfo_t* gpuFabricInfo);
}  // namespace wholememory

#endif
