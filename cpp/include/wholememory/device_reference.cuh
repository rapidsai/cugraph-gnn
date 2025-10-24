/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <assert.h>
#include <cstddef>

#include "global_reference.h"
namespace wholememory {

template <typename DataTypeT>
class device_reference {
 public:
  __device__ __forceinline__ explicit device_reference(const wholememory_gref_t& gref)
    : pointer_(static_cast<DataTypeT*>(gref.pointer)),
      typed_stride_(gref.stride / sizeof(DataTypeT)),
      world_size_(gref.world_size),
      same_chunk_(gref.same_chunk)
  {
    assert(gref.stride % sizeof(DataTypeT) == 0);
    if (typed_stride_ != 0 && !same_chunk_) {
      assert(world_size_ <= 8);  // intra-node WHOLEMEMORY_MT_CHUNKED
      for (int i = 0; i < world_size_ + 1; i++) {
        assert(gref.rank_memory_offsets[i] % sizeof(DataTypeT) == 0);
        typed_rank_mem_offsets_[i] = gref.rank_memory_offsets[i] / sizeof(DataTypeT);
      }
    }
  }
  __device__ device_reference() = delete;

  __device__ __forceinline__ DataTypeT& operator[](size_t index)
  {
    if (typed_stride_ == 0) { return pointer_[index]; }
    if (same_chunk_) {
      size_t rank = index / typed_stride_;
      return static_cast<DataTypeT**>(
        static_cast<void*>(pointer_))[rank][index - rank * typed_stride_];
    } else {
      size_t rank = 0;
      for (int i = 1; i < world_size_ + 1; i++) {
        if (index < typed_rank_mem_offsets_[i]) {
          rank = i - 1;
          break;
        }
      }
      return static_cast<DataTypeT**>(
        static_cast<void*>(pointer_))[rank][index - typed_rank_mem_offsets_[rank]];
    }
  }

 private:
  DataTypeT* pointer_;
  int world_size_;
  size_t typed_stride_;

  bool same_chunk_;
  size_t typed_rank_mem_offsets_[8 + 1];
};

}  // namespace wholememory
