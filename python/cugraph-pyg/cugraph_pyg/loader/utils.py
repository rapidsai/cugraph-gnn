# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from cugraph_pyg.utils.imports import import_optional

torch = import_optional("torch")


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
