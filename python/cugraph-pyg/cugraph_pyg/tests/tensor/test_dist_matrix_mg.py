# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import os
import pytest
import torch
from cugraph_pyg.tensor import DistMatrix

from pylibwholegraph.torch.initialize import init as wm_init
from pylibwholegraph.binding.wholememory_binding import finalize as wm_finalize


def run_test_dist_matrix_creation(rank, world_size, device):
    """Test basic DistMatrix creation from tensors"""
    torch.cuda.set_device(rank)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    wm_init(rank, world_size, rank, world_size)

    # Create test data
    if rank == 0:
        col = torch.randint(0, 100, (1000,), dtype=torch.long, device="cuda")
        row = torch.randint(0, 100, (1000,), dtype=torch.long, device="cuda")
    else:
        col = torch.zeros(1000, dtype=torch.long, device="cuda")
        row = torch.zeros(1000, dtype=torch.long, device="cuda")
    torch.distributed.broadcast(col, src=0)
    torch.distributed.broadcast(row, src=0)

    # Create distributed matrix
    dist_matrix = DistMatrix(src=(col, row), device=device, format="coo")

    assert dist_matrix.shape == (1000, 1000)
    assert dist_matrix.dtype == torch.long
    assert dist_matrix._format == "coo"

    torch.distributed.barrier()

    # Test __getitem__
    idx = torch.randint(0, 1000, (10,))
    result = dist_matrix[idx]
    assert result.shape == (2, 10)
    assert torch.allclose(result[0], col[idx])
    assert torch.allclose(result[1], row[idx])

    torch.distributed.barrier()

    wm_finalize()
    torch.distributed.destroy_process_group()


def run_test_dist_matrix_empty_creation(rank, world_size, device):
    """Test DistMatrix creation with empty initialization"""
    torch.cuda.set_device(rank)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    wm_init(rank, world_size, rank, world_size)

    # Create empty distributed matrix
    dist_matrix = DistMatrix(
        shape=(100, 100), dtype=torch.long, device=device, format="coo"
    )

    assert dist_matrix.shape == (100, 100)
    assert dist_matrix.dtype == torch.long
    assert dist_matrix._format == "coo"

    # Test setting values
    if rank == 0:
        perm = torch.randperm(100, dtype=torch.long, device="cuda")
        val_col = torch.randint(0, 100, (100,), dtype=torch.long, device="cuda")
        val_row = torch.randint(0, 100, (100,), dtype=torch.long, device="cuda")
    else:
        perm = torch.empty((100,), dtype=torch.long, device="cuda")
        val_col = torch.empty((100,), dtype=torch.long, device="cuda")
        val_row = torch.empty((100,), dtype=torch.long, device="cuda")
    torch.distributed.broadcast(perm, src=0)
    torch.distributed.broadcast(val_col, src=0)
    torch.distributed.broadcast(val_row, src=0)

    # Test __setitem__ with tuple
    perm_local = torch.tensor_split(perm, world_size)[rank]
    val_col_local = torch.tensor_split(val_col, world_size)[rank]
    val_row_local = torch.tensor_split(val_row, world_size)[rank]
    dist_matrix[perm_local] = (val_col_local, val_row_local)

    # Verify values were set correctly
    out = dist_matrix[perm]
    assert torch.allclose(out[0], val_col)
    assert torch.allclose(out[1], val_row)

    wm_finalize()
    torch.distributed.destroy_process_group()


def run_test_dist_matrix_invalid_cases(rank, world_size, device):
    """Test DistMatrix creation with invalid cases"""
    torch.cuda.set_device(rank)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    wm_init(rank, world_size, rank, world_size)

    # Test invalid format
    with pytest.raises(ValueError):
        DistMatrix(shape=(100, 100), dtype=torch.long, format="invalid", device=device)

    # Test missing required parameters
    with pytest.raises(ValueError):
        DistMatrix(device=device)  # Missing both shape and src

    # Test invalid source type
    with pytest.raises(ValueError):
        DistMatrix(src={}, device=device)  # Invalid source type

    # Test invalid source tuple length
    col = torch.randint(0, 100, (100,), dtype=torch.long, device="cuda")
    with pytest.raises(ValueError):
        DistMatrix(src=(col,), device=device)  # Invalid tuple length

    # Test shape mismatch in COO format
    col = torch.randint(0, 100, (100,), dtype=torch.long, device="cuda")
    row = torch.randint(0, 100, (200,), dtype=torch.long, device="cuda")
    with pytest.raises(ValueError):
        DistMatrix(src=(col, row), format="coo", device=device)

    wm_finalize()
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dist_matrix(device):
    """Run all DistMatrix tests"""

    world_size = torch.cuda.device_count()

    # simulate torchrun call
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    for func in [
        run_test_dist_matrix_creation,
        run_test_dist_matrix_empty_creation,
        run_test_dist_matrix_invalid_cases,
    ]:
        torch.multiprocessing.spawn(
            func,
            args=(world_size, device),
            nprocs=world_size,
        )
