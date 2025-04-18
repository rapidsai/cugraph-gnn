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

import os
import pytest
import tempfile

from cugraph_pyg.tensor import DistTensor, DistEmbedding
from cugraph.utilities.utils import import_optional, MissingModule
from pylibwholegraph.torch.initialize import init as wm_init
from pylibwholegraph.binding.wholememory_binding import finalize as wm_finalize

torch = import_optional("torch")
pylibwholegraph = import_optional("pylibwholegraph")


def run_test_dist_tensor_creation(rank, world_size, device, clx):
    """Test basic DistTensor creation from a tensor"""
    torch.cuda.set_device(rank)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    wm_init(rank, world_size, rank, world_size)

    # Create a distributed tensor
    if rank == 0:
        features = torch.randn(
            world_size * 100 * 10, dtype=torch.float32, device="cuda"
        )
        torch.distributed.broadcast(features, src=0)
    else:
        features = torch.zeros(
            world_size * 100 * 10, dtype=torch.float32, device="cuda"
        )
        torch.distributed.broadcast(features, src=0)
    torch.distributed.barrier()  # helps catch errors

    features = features.reshape((-1, 10)).to(device)

    dist_tensor = clx.from_tensor(tensor=features, device=device)
    assert dist_tensor.shape == features.shape
    assert dist_tensor.dtype == features.dtype
    assert dist_tensor.device == device

    torch.distributed.barrier()  # helps catch errors

    ix = torch.randint(0, features.shape[0], (10,))
    assert torch.allclose(features[ix].cpu(), dist_tensor[ix].cpu())

    wm_finalize()
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    isinstance(pylibwholegraph, MissingModule), reason="wholegraph not available"
)
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("clx", [DistTensor, DistEmbedding])
@pytest.mark.mg
def test_dist_tensor_creation(device, clx):
    world_size = torch.cuda.device_count()

    # simulate torchrun call
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    torch.multiprocessing.spawn(
        run_test_dist_tensor_creation,
        args=(world_size, device, clx),
        nprocs=world_size,
    )


def run_test_dist_tensor_from_file(rank, world_size, device, clx):
    """Test DistTensor creation from file"""
    torch.cuda.set_device(rank)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    wm_init(rank, world_size, rank, world_size)

    # Create test data
    features = torch.arange(0, world_size * 1000)
    features = features.reshape((features.numel() // 100, 100)).to(torch.float32)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(features, f.name)
        file_path = f.name

    # Load distributed tensor
    torch.distributed.barrier()
    print(f"loading from {file_path}...")
    dist_tensor = clx.from_file(file_path, device=device)
    print("loaded...")
    assert dist_tensor.shape == features.shape
    assert dist_tensor.dtype == features.dtype
    assert dist_tensor.device == device

    ix = torch.randperm(features.shape[0])[:10]
    assert torch.allclose(features[ix].cpu(), dist_tensor[ix].cpu())

    torch.distributed.barrier()
    # Clean up
    if rank == 0:
        os.unlink(file_path)

    wm_finalize()
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    isinstance(pylibwholegraph, MissingModule), reason="wholegraph not available"
)
@pytest.mark.skipif(isinstance(torch, MissingModule), reason="torch not available")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("clx", [DistTensor, DistEmbedding])
@pytest.mark.mg
def test_dist_tensor_from_file(device, clx):
    world_size = torch.cuda.device_count()

    # simulate torchrun call
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)

    torch.multiprocessing.spawn(
        run_test_dist_tensor_from_file,
        args=(world_size, device, clx),
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
