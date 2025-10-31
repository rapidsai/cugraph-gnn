/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cuda_macros.hpp"

#include <algorithm>
#include <cstdlib>

#include <cuda_runtime_api.h>

namespace wholememory {

static bool s_debug_sync_mode = false;

__attribute__((constructor)) static void ReadDebugSyncModeFromEnv()
{
  try {
    char* debug_sync_env_str = std::getenv("WM_DEBUG_SYNC");
    if (debug_sync_env_str != nullptr) {
      std::string str = debug_sync_env_str;
      std::transform(
        str.begin(), str.end(), str.begin(), [](unsigned char c) { return std::tolower(c); });
      if (str == "1" || str == "on" || str == "true") {
        printf("[Notice] Enabled Debug Sync, performance may suffer.\n");
        s_debug_sync_mode = true;
      }
    }
  } catch (...) {
    return;
  }
}

void set_debug_sync_mode(bool debug_sync_mode) { s_debug_sync_mode = debug_sync_mode; }

void debug_synchronize(const char* filename, int line, cudaStream_t stream)
{
  if (s_debug_sync_mode) {
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
      printf("CUDA cudaGetLastError() failed at file=%s line=%d failed with %s\n",
             filename,
             line,
             cudaGetErrorString(status));
      abort();
    }
    status = cudaStreamSynchronize(stream);
    if (status != cudaSuccess) {
      printf("CUDA cudaStreamSynchronize() failed at file=%s line=%d failed with %s\n",
             filename,
             line,
             cudaGetErrorString(status));
      abort();
    }
  }
}

}  // namespace wholememory
