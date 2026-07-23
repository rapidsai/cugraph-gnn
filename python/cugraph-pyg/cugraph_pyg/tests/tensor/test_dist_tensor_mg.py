# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pytest
import tempfile

from cugraph_pyg.tensor import DistTensor, DistEmbedding
from cugraph_pyg.utils.imports import import_optional, MissingModule
from pylibwholegraph.torch.initialize import init as wm_init
from pylibwholegraph.binding.wholememory_binding import finalize as wm_finalize

torch = import_optional("torch")
pylibwholegraph = import_optional("pylibwholegraph")


def run_test_dist_tensor_creation(rank, world_size, device, clx, dtype):
    """Test basic DistTensor creation from a tensor"""
    torch.cuda.set_device(rank)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    wm_init(rank, world_size, rank, world_size)

    dtype = getattr(torch, dtype)

    # Create a distributed tensor
    if rank == 0:
        features = torch.randn(world_size * 100 * 10, dtype=dtype, device="cuda")
        torch.distributed.broadcast(features, src=0)
    else:
        features = torch.zeros(world_size * 100 * 10, dtype=dtype, device="cuda")
        torch.distributed.broadcast(features, src=0)
    torch.distributed.barrier()  # helps catch errors

    features = features.reshape((-1, 10)).to(device)

    dist_tensor = clx.from_tensor(tensor=features, device=device)
    assert dist_tensor.shape == features.shape
    assert dist_tensor.dtype == features.dtype
    assert dist_tensor.device == device

    torch.distributed.barrier()  # helps catch errors

    ix = torch.randint(0, features.shape[0], (10,))
    assert torch.allclose(features[ix].cpu().float(), dist_tensor[ix].cpu().float())

    wm_finalize()
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    isinstance(pylibwholegraph, MissingModule), reason="wholegraph not available"
)
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("clx", [DistTensor, DistEmbedding])
@pytest.mark.mg
def test_dist_tensor_creation(device, clx, dtype):
    world_size = torch.cuda.device_count()

    # simulate torchrun call
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    torch.multiprocessing.spawn(
        run_test_dist_tensor_creation,
        args=(world_size, device, clx, dtype),
        nprocs=world_size,
    )


def run_test_dist_tensor_from_file(rank, world_size, device, clx, dtype, file_format):
    """Test DistTensor creation from file"""
    torch.cuda.set_device(rank)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    wm_init(rank, world_size, rank, world_size)

    dtype = getattr(torch, dtype)

    # Create test data
    features = torch.arange(0, world_size * 1000)
    features = features.reshape((features.numel() // 100, 100)).to(dtype)

    suffix = ".pt" if file_format == "pytorch" else ".parquet"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        file_path = f.name
    if file_format == "pytorch":
        torch.save(features, file_path)
    else:
        import pyarrow
        import pyarrow.parquet as parquet

        parquet.write_table(
            pyarrow.table(
                {
                    f"feature_{index}": features[:, index].numpy()
                    for index in range(features.shape[1])
                }
            ),
            file_path,
        )

    # Load distributed tensor
    torch.distributed.barrier()
    print(f"loading from {file_path}...")
    if file_format == "pytorch":
        dist_tensor = clx.from_file(file_path, device=device)
    else:
        dist_tensor = clx.from_file(
            file_path,
            device=device,
            shape=features.shape,
            dtype=features.dtype,
            file_format="parquet",
        )
    print("loaded...")
    assert dist_tensor.shape == features.shape
    assert dist_tensor.dtype == features.dtype
    assert dist_tensor.device == device

    ix = torch.randperm(features.shape[0])[:10]
    assert torch.allclose(features[ix].cpu().float(), dist_tensor[ix].cpu().float())

    torch.distributed.barrier()
    # Clean up
    os.unlink(file_path)

    wm_finalize()
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    isinstance(pylibwholegraph, MissingModule), reason="wholegraph not available"
)
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("clx", [DistTensor, DistEmbedding])
@pytest.mark.parametrize("file_format", ["pytorch", "parquet"])
@pytest.mark.mg
def test_dist_tensor_from_file(device, clx, dtype, file_format):
    if file_format == "parquet" and dtype != "float32":
        pytest.skip("Parquet integration coverage uses a float32 source")
    world_size = torch.cuda.device_count()

    # simulate torchrun call
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    torch.multiprocessing.spawn(
        run_test_dist_tensor_from_file,
        args=(world_size, device, clx, dtype, file_format),
        nprocs=world_size,
    )


def run_test_dist_tensor_invalid_cases(rank, world_size):
    """Test DistTensor creation with invalid cases"""
    torch.cuda.set_device(rank)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    wm_init(rank, world_size, rank, world_size)

    # Test invalid shape
    with pytest.raises(ValueError):
        DistTensor(shape=[1, 2, 3])  # 3D shape not supported

    # Test missing required parameters
    with pytest.raises(ValueError):
        DistTensor()  # Missing both shape and src

    # Test invalid source
    with pytest.raises(ValueError):
        DistTensor(src="invalid.txt")  # Unsupported file format

    wm_finalize()
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    isinstance(pylibwholegraph, MissingModule), reason="wholegraph not available"
)
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.mg
def test_dist_tensor_invalid_cases():
    world_size = torch.cuda.device_count()

    # simulate torchrun call
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    torch.multiprocessing.spawn(
        run_test_dist_tensor_invalid_cases,
        args=(world_size,),
        nprocs=world_size,
    )
