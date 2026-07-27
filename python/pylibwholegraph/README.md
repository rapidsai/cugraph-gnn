<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# pylibwholegraph

WholeGraph supports PyTorch and provides a distributed graph and kv store.

cuGraph-PyG can leverage WholeGraph for even greater scalability.

## Using an RMM memory resource

WholeGraph can use RMM's current per-device memory resource for distributed and hierarchy tensors
stored on the device. Configure RMM after selecting the process's CUDA device, then enable RMM
allocation before creating WholeMemory tensors:

```python
import rmm
import pylibwholegraph.torch as wgth

rmm.reinitialize(pool_allocator=True, initial_pool_size=2**30)
wgth.set_rmm_enabled(True)
```

WholeMemory looks up RMM's current resource when each supported allocation is created, so normal
RMM per-device resource configuration applies. Call `wgth.set_rmm_enabled(False)` to use
WholeMemory's default allocator for future allocations.

Chunked, continuous, and NVSHMEM device tensors require specialized CUDA allocation mechanisms.
When RMM is enabled, these tensors emit a warning and use their existing allocation path.
