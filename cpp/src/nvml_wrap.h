// Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#if CUDA_VERSION >= 12030

#pragma once

#include <nvml.h>

bool NvmlFabricSymbolLoaded();

typedef nvmlReturn_t (*nvmlDeviceGetHandleByIndexFunc)(unsigned int, nvmlDevice_t*);
typedef nvmlReturn_t (*nvmlDeviceGetGpuFabricInfoFunc)(nvmlDevice_t, nvmlGpuFabricInfo_t*);

extern nvmlDeviceGetHandleByIndexFunc nvmlDeviceGetHandleByIndexPtr;
extern nvmlDeviceGetGpuFabricInfoFunc nvmlDeviceGetGpuFabricInfoPtr;

#endif  // CUDA_VERSION >= 12030
