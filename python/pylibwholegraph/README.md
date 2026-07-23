<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# pylibwholegraph

WholeGraph supports PyTorch and provides a distributed graph and kv store.

cuGraph-PyG can leverage WholeGraph for even greater scalability.

## Using an RMM memory resource

WholeGraph can use an RMM memory resource for distributed and hierarchy tensors stored on the
device. Configure the resource after selecting the process's CUDA device and before creating
WholeMemory tensors:

```python
import rmm
import pylibwholegraph.torch as wgth

memory_resource = rmm.mr.PoolMemoryResource(
    rmm.mr.CudaMemoryResource(),
    initial_pool_size=2**30,
)
wgth.set_memory_resource(memory_resource)
```

This also installs the resource as RMM's current resource for the active device. Pass `None` to
`set_memory_resource()` to disable RMM for future WholeMemory allocations.

Chunked, continuous, and NVSHMEM device tensors require specialized CUDA allocation mechanisms.
When RMM is enabled, these tensors emit a warning and use their existing allocation path.
