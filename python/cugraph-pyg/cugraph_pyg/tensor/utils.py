# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from typing import Union, List

import os

from cugraph_pyg.utils.imports import import_optional

torch = import_optional("torch")
wgth = import_optional("pylibwholegraph.torch")


def copy_host_global_tensor_to_local(wm_tensor, host_tensor, wm_comm):
    local_tensor, local_start = wm_tensor.get_local_tensor(host_view=False)

    local_tensor.copy_(host_tensor[local_start : local_start + local_tensor.shape[0]])
    wm_comm.barrier()


def create_wg_dist_tensor(
    shape: list,
    dtype: "torch.dtype",
    location: str = "cpu",
    partition_book: Union[List[int], None] = None,
    backend: str = "nccl",
    **kwargs,
):
    """
    Create a WholeGraph-managed distributed tensor.

    Parameters
    ----------
    shape : list
        The shape of the tensor. It has to be a two-dimensional
        or one-dimensional tensor for now.
        The first dimension typically is the number of nodes/edges.
        The second dimension is the feature/embedding dimension.
    dtype : torch.dtype
        The dtype of the tensor.
    location : str, optional
        The desired location to store the embedding [ "cpu" | "cuda" ]
    partition_book : list, optional
        The partition book for the embedding tensor.
        The length of the partition book should be the same as the number of ranks.
        Defaults to an even partitioning scheme.
    backend : str, optional
        The backend for the distributed tensor [ "nccl" | "vmm" | "nvshmem" ]
    """
    global_comm = wgth.get_global_communicator()

    if backend == "nccl":
        embedding_wholememory_type = "distributed"
    elif backend == "vmm":
        embedding_wholememory_type = "continuous"
    elif backend == "nvshmem":
        raise NotImplementedError("NVSHMEM backend is not implemented in cuGraph-PyG.")
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    embedding_wholememory_location = location

    if "cache_policy" in kwargs:
        if len(shape) != 2:
            raise ValueError("The shape of the embedding tensor must be 2D.")

        cache_policy = kwargs["cache_policy"]
        kwargs.pop("cache_policy")

        wm_embedding = wgth.create_embedding(
            global_comm,
            embedding_wholememory_type,
            embedding_wholememory_location,
            dtype,
            shape,
            cache_policy=cache_policy,
            embedding_entry_partition=partition_book,
            **kwargs,
        )
    else:
        if len(shape) not in [1, 2]:
            raise ValueError("The shape of the tensor must be 2D or 1D.")

        wm_embedding = wgth.create_wholememory_tensor(
            global_comm,
            embedding_wholememory_type,
            embedding_wholememory_location,
            shape,
            dtype,
            strides=None,
            tensor_entry_partition=partition_book,
        )

    return wm_embedding


def create_wg_dist_tensor_from_files(
    file_list: List[str],
    shape: list,
    dtype: torch.dtype,
    location: str = "cpu",
    partition_book: Union[List[int], None] = None,
    backend: str = "nccl",
    **kwargs,
):
    """
    Create a WholeGraph-managed distributed tensor from a list of files.

    Parameters
    ----------
    file_list : list
        The list of files to load the embedding tensor.
    shape : list
        The shape of the tensor. It has to be a two-dimensional
        or one-dimensional tensor for now.
        The first dimension typically is the number of nodes/edges.
        The second dimension is the feature/embedding dimension.
    dtype : torch.dtype
        The dtype of the tensor.
    location : str, optional
        The desired location to store the embedding [ "cpu" | "cuda" ]
    partition_book : list, optional
        The partition book for the embedding tensor.
        The length of the partition book should be the same as the number of ranks.
        Defaults to an even partitioning scheme.
    backend : str, optional
        The backend for the distributed tensor [ "nccl" | "vmm" | "nvshmem" ]
    """
    global_comm = wgth.get_global_communicator()

    if backend == "nccl":
        embedding_wholememory_type = "distributed"
    elif backend == "vmm":
        embedding_wholememory_type = "continuous"
    elif backend == "nvshmem":
        raise NotImplementedError("NVSHMEM backend is not implemented in cuGraph-PyG.")
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    embedding_wholememory_location = location

    if "cache_policy" in kwargs:
        assert len(shape) == 2, "The shape of the embedding tensor must be 2D."
        cache_policy = kwargs["cache_policy"]
        kwargs.pop("cache_policy")

        wm_embedding = wgth.create_embedding_from_filelist(
            global_comm,
            embedding_wholememory_type,
            embedding_wholememory_location,
            file_list,
            dtype,
            shape[1],
            cache_policy=cache_policy,
            embedding_entry_partition=partition_book,
            **kwargs,
        )
    else:
        assert len(shape) == 2 or len(shape) == 1, (
            "The shape of the tensor must be 2D or 1D."
        )
        last_dim_size = 0 if len(shape) == 1 else shape[1]
        wm_embedding = wgth.create_wholememory_tensor_from_filelist(
            global_comm,
            embedding_wholememory_type,
            embedding_wholememory_location,
            file_list,
            dtype,
            last_dim_size,
            tensor_entry_partition=partition_book,
        )
    return wm_embedding


def has_nvlink_network():
    r"""Check if the current hardware supports cross-node NVLink network."""

    global_comm = wgth.comm.get_global_communicator("nccl")
    local_size = int(os.environ["LOCAL_WORLD_SIZE"])

    world_size = torch.distributed.get_world_size()

    # Intra-node communication
    if local_size == world_size:
        # use WholeGraph to check if the current hardware supports direct p2p
        return global_comm.support_type_location("continuous", "cuda")

    # Check for multi-node support
    is_cuda_supported = global_comm.support_type_location("continuous", "cuda")
    is_cpu_supported = global_comm.support_type_location("continuous", "cpu")

    if is_cuda_supported and is_cpu_supported:
        return True

    return False


def is_empty(a):
    return a.numel() == 0


def empty(dim: int = 1):
    if dim == 1:
        return torch.tensor([], dtype=torch.int32)
    elif dim == 2:
        return torch.tensor([], dtype=torch.int32).view(0, 1)
    else:
        raise ValueError(f"Unsupported dimension: {dim}")
