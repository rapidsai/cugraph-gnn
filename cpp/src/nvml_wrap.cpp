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
#include "nvml_wrap.h"

#if CUDA_VERSION >= 12030
#include <dlfcn.h>
#include <mutex>
#include <stdio.h>

namespace {

void* nvml_handle = nullptr;
std::mutex nvml_mutex;
bool nvml_loaded = false;

bool LoadNvmlLibrary()
{
  nvml_handle = dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (!nvml_handle) {
    nvml_handle = dlopen("libnvidia-ml.so", RTLD_NOW);
    if (!nvml_handle) {
      fprintf(stderr, "Failed to load NVML library: %s\n", dlerror());
      return false;
    }
  }
  return true;
}

template <typename T>
T LoadNvmlSymbol(const char* name)
{
  void* symbol = dlsym(nvml_handle, name);
  if (!symbol) { return nullptr; }
  return reinterpret_cast<T>(symbol);
}

}  // namespace

// Global function pointers
nvmlDeviceGetHandleByIndexFunc nvmlDeviceGetHandleByIndexPtr = nullptr;
nvmlDeviceGetGpuFabricInfoFunc nvmlDeviceGetGpuFabricInfoPtr = nullptr;

// Ensure NVML is loaded and symbols are initialized
bool NvmlFabricSymbolLoaded()
{
  std::lock_guard<std::mutex> lock(nvml_mutex);
  if (nvml_loaded) {
    return true;  // Already loaded
  }

  if (LoadNvmlLibrary()) {
    nvmlDeviceGetHandleByIndexPtr =
      LoadNvmlSymbol<nvmlDeviceGetHandleByIndexFunc>("nvmlDeviceGetHandleByIndex");
    nvmlDeviceGetGpuFabricInfoPtr =
      LoadNvmlSymbol<nvmlDeviceGetGpuFabricInfoFunc>("nvmlDeviceGetGpuFabricInfo");

    if (!nvmlDeviceGetHandleByIndexPtr || !nvmlDeviceGetGpuFabricInfoPtr) {
      dlclose(nvml_handle);
      nvml_handle = nullptr;
    } else {
      nvml_loaded = true;
    }
  }
  return nvml_loaded;
}
#endif  // CUDA_VERSION >= 12030
