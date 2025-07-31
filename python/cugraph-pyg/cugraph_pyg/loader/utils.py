# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch


def generate_seed():
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    if rank == 0:
        seed = torch.randint(
            0, 2**63 - world_size, (1,), dtype=torch.int64, device="cuda"
        )
    else:
        seed = torch.tensor([0], dtype=torch.int64, device="cuda")
    torch.distributed.broadcast(seed, src=0)
    seed = seed.item() + rank
    return seed
