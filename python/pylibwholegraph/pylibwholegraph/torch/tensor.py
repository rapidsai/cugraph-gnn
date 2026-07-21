# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pylibwholegraph.binding.wholememory_binding as wmb
from pylibwholegraph.utils.imports import import_optional
from .utils import (
    torch_dtype_to_wholememory_dtype,
    wholememory_dtype_to_torch_dtype,
    get_file_size,
)
from .utils import str_to_wmb_wholememory_memory_type, str_to_wmb_wholememory_location
from .utils import get_part_file_name, get_part_file_list
from .comm import WholeMemoryCommunicator
from typing import Union, List
from .dlpack_utils import torch_import_from_dlpack
from .wholegraph_env import wrap_torch_tensor, get_wholegraph_env_fns, get_stream

torch = import_optional("torch")
np = import_optional("numpy")
pq = import_optional("pyarrow.parquet")

WholeMemoryMemoryType = wmb.WholeMemoryMemoryType
WholeMemoryMemoryLocation = wmb.WholeMemoryMemoryLocation


_FILE_FORMAT_ALIASES = {
    "bin": "binary",
    "raw": "binary",
    "binary": "binary",
    "pt": "pytorch",
    "pth": "pytorch",
    "torch": "pytorch",
    "pytorch": "pytorch",
    "parquet": "parquet",
    "pq": "parquet",
    "auto": "auto",
}


def _normalize_filelist(filelist: Union[List[str], str]) -> List[str]:
    if isinstance(filelist, str):
        normalized_filelist = [filelist]
    else:
        normalized_filelist = list(filelist)
    if not normalized_filelist:
        raise ValueError("filelist must contain at least one file")
    return normalized_filelist


def _resolve_file_format(filelist: List[str], file_format: str) -> str:
    try:
        normalized_format = _FILE_FORMAT_ALIASES[file_format.lower()]
    except (AttributeError, KeyError):
        raise ValueError(
            "file_format must be one of 'binary', 'pytorch', 'parquet', or 'auto'"
        ) from None

    if normalized_format != "auto":
        return normalized_format

    detected_formats = set()
    for filename in filelist:
        extension = os.path.splitext(os.fspath(filename))[1].lower()
        if extension in (".pt", ".pth"):
            detected_formats.add("pytorch")
        elif extension in (".parquet", ".pq"):
            detected_formats.add("parquet")
        else:
            detected_formats.add("binary")

    if len(detected_formats) != 1:
        raise ValueError("All files must have the same format when file_format='auto'")
    return detected_formats.pop()


def _load_pytorch_tensor(filename: str):
    value = torch.load(filename, map_location="cpu", weights_only=True)
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        tensors = [item for item in value.values() if isinstance(item, torch.Tensor)]
        if len(tensors) == 1:
            return tensors[0].detach()
    raise ValueError(
        f"PyTorch file {filename!r} must contain a tensor or exactly one tensor "
        "in a dictionary"
    )


def _load_parquet_tensor(filename: str):
    table = pq.read_table(filename)
    if table.num_columns == 0:
        raise ValueError(f"Parquet file {filename!r} contains no columns")
    try:
        columns = [
            column.combine_chunks().to_numpy(zero_copy_only=False)
            for column in table.columns
        ]
        return torch.as_tensor(np.column_stack(columns))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Parquet file {filename!r} must contain only numeric columns"
        ) from error


def _load_structured_tensor(
    filename: str,
    file_format: str,
    dtype: "torch.dtype",
    dim: int,
    last_dim_size: int,
):
    if file_format == "pytorch":
        tensor = _load_pytorch_tensor(filename)
    elif file_format == "parquet":
        tensor = _load_parquet_tensor(filename)
    else:
        raise ValueError(f"Unsupported structured file format {file_format!r}")

    if dim == 1:
        if tensor.dim() == 2 and tensor.shape[1] == 1:
            tensor = tensor[:, 0]
        if tensor.dim() != 1:
            raise ValueError(
                f"File {filename!r} has shape {tuple(tensor.shape)}, expected a "
                "1-D tensor"
            )
    elif tensor.dim() != 2 or tensor.shape[1] != last_dim_size:
        raise ValueError(
            f"File {filename!r} has shape {tuple(tensor.shape)}, expected shape "
            f"(N, {last_dim_size})"
        )

    return tensor.to(device="cpu", dtype=dtype).contiguous()


def _get_filelist_entry_count(
    filelist: List[str],
    file_format: str,
    dtype: "torch.dtype",
    last_dim_size: int,
) -> int:
    if file_format == "binary":
        element_size = torch.tensor([], dtype=dtype).element_size()
        file_entry_size = (
            element_size * last_dim_size if last_dim_size > 0 else element_size
        )
        total_file_size = 0
        for filename in filelist:
            file_size = get_file_size(filename)
            if file_size % file_entry_size != 0:
                raise ValueError(
                    "File %s size is %d not multiple of %d"
                    % (filename, file_size, file_entry_size)
                )
            total_file_size += file_size
        return total_file_size // file_entry_size

    dim = 2 if last_dim_size > 0 else 1
    return sum(
        _load_structured_tensor(filename, file_format, dtype, dim, last_dim_size).shape[
            0
        ]
        for filename in filelist
    )


class WholeMemoryTensor(object):
    r"""WholeMemory Tensor"""

    def __init__(self, wmb_tensor: wmb.PyWholeMemoryTensor):
        self.wmb_tensor = wmb_tensor

    @property
    def dtype(self):
        return wholememory_dtype_to_torch_dtype(self.wmb_tensor.dtype)

    def dim(self):
        return self.wmb_tensor.dim()

    @property
    def shape(self):
        return self.wmb_tensor.shape

    def stride(self):
        return self.wmb_tensor.stride()

    def storage_offset(self):
        return self.wmb_tensor.storage_offset()

    def get_comm(self):
        return WholeMemoryCommunicator(
            self.wmb_tensor.get_wholememory_handle().get_communicator()
        )

    def gather(
        self, indice: "torch.Tensor", *, force_dtype: Union["torch.dtype", None] = None
    ):
        assert indice.dim() == 1
        embedding_dim = self.shape[1] if self.dim() == 2 else 1
        embedding_count = indice.shape[0]
        current_cuda_device = "cuda:%d" % (torch.cuda.current_device(),)
        output_dtype = force_dtype if force_dtype is not None else self.dtype
        output_tensor = torch.empty(
            [embedding_count, embedding_dim],
            device=current_cuda_device,
            dtype=output_dtype,
            requires_grad=False,
        )
        wmb.wholememory_gather_op(
            self.wmb_tensor,
            wrap_torch_tensor(indice),
            wrap_torch_tensor(output_tensor),
            get_wholegraph_env_fns(),
            get_stream(),
        )
        return output_tensor.view(-1) if self.dim() == 1 else output_tensor

    def scatter(self, input_tensor: "torch.Tensor", indice: "torch.Tensor"):
        assert indice.dim() == 1
        assert input_tensor.dim() == self.dim()
        assert indice.shape[0] == input_tensor.shape[0]
        if self.dim() == 2:
            assert input_tensor.shape[1] == self.shape[1]
        else:
            # unsqueeze to 2D tensor because wmb_tensor is unsqueezed within scatter_op
            input_tensor = input_tensor.unsqueeze(1)
        wmb.wholememory_scatter_op(
            wrap_torch_tensor(input_tensor),
            wrap_torch_tensor(indice),
            self.wmb_tensor,
            get_wholegraph_env_fns(),
            get_stream(),
        )

    def get_sub_tensor(self, starts, ends):
        """
        Get sub tensor of WholeMemory Tensor
        :param starts: An array of the start indices of each dim
        :param ends: An array of the end indices of each dim, -1 means
          to the last element
        :return: WholeMemory Tensor
        """
        return WholeMemoryTensor(self.wmb_tensor.get_sub_tensor(starts, ends))

    def get_local_tensor(self, host_view: bool = False):
        """Get local tensor of WholeMemory Tensor
        :param host_view: Get host view or not, if True, return host tensor,
          else return device tensor
        :return: Tuple of DLPack Tensor and element offset.
        """
        if host_view:
            return self.wmb_tensor.get_local_tensor(
                torch_import_from_dlpack, WholeMemoryMemoryLocation.MlHost, -1
            )
        else:
            return self.wmb_tensor.get_local_tensor(
                torch_import_from_dlpack,
                WholeMemoryMemoryLocation.MlDevice,
                torch.cuda.current_device(),
            )

    def get_global_tensor(self, host_view: bool = False):
        """Get global tensor of WholeMemory Tensor
        :param host_view: Get host view or not, if True, return host tensor,
          else return device tensor
        :return: Tuple of DLPack Tensor and element offset (0 for global tensor).
        """
        if host_view:
            return self.wmb_tensor.get_global_tensor(
                torch_import_from_dlpack, WholeMemoryMemoryLocation.MlHost, -1
            )
        else:
            return self.wmb_tensor.get_global_tensor(
                torch_import_from_dlpack,
                WholeMemoryMemoryLocation.MlDevice,
                torch.cuda.current_device(),
            )

    def get_all_chunked_tensor(self, host_view: bool = False):
        """Get all chunked tensor of WholeMemory Tensor
        :param host_view: Get host view or not, if True, return host tensor,
          else return device tensor
        :return: Tuple of DLPack Tensors and element offsets.
        """
        if host_view:
            return self.wmb_tensor.get_global_tensorget_all_chunked_tensor(
                torch_import_from_dlpack, WholeMemoryMemoryLocation.MlHost, -1
            )
        else:
            return self.wmb_tensor.get_global_tensorget_all_chunked_tensor(
                torch_import_from_dlpack,
                WholeMemoryMemoryLocation.MlDevice,
                torch.cuda.current_device(),
            )

    def from_filelist(
        self,
        filelist: Union[List[str], str],
        round_robin_size: int = 0,
        file_format: str = "binary",
    ):
        """
        Load WholeMemory Tensor from file lists
        :param filelist: file list to load from
        :param round_robin_size: continuous embedding size of a rank
          using round robin shard strategy
        :param file_format: file format, one of binary, pytorch, parquet, or auto.
          PyTorch files must contain a tensor or a dictionary with exactly one
          tensor. Parquet files must contain only numeric columns.
        :return: None
        """
        filelist = _normalize_filelist(filelist)
        file_format = _resolve_file_format(filelist, file_format)
        if file_format == "binary":
            self.wmb_tensor.from_filelist(filelist, round_robin_size)
            return

        if round_robin_size != 0:
            raise ValueError(
                "round_robin_size is only supported for binary file loading"
            )

        local_tensor, local_start = self.get_local_tensor()
        local_end = local_start + local_tensor.shape[0]
        file_start = 0
        last_dim_size = self.shape[1] if self.dim() == 2 else 0

        for filename in filelist:
            file_tensor = _load_structured_tensor(
                filename, file_format, self.dtype, self.dim(), last_dim_size
            )
            file_end = file_start + file_tensor.shape[0]
            copy_start = max(file_start, local_start)
            copy_end = min(file_end, local_end)
            if copy_start < copy_end:
                source_start = copy_start - file_start
                destination_start = copy_start - local_start
                copy_count = copy_end - copy_start
                local_tensor[destination_start : destination_start + copy_count].copy_(
                    file_tensor[source_start : source_start + copy_count]
                )
            file_start = file_end

        if file_start != self.shape[0]:
            raise ValueError(
                f"Files contain {file_start} entries, but the WholeMemory tensor "
                f"expects {self.shape[0]} entries"
            )
        torch.cuda.synchronize()
        self.get_comm().barrier()

    def from_file_prefix(self, file_prefix: str, part_count: Union[int, None] = None):
        """
        Load WholeMemory  tensor from files with same prefix, files has format
            "%s_part_%d_of_%d" % (prefix, part_id, part_count)
        :param file_prefix: file name prefix
        :param part_count: part count of file
        :return: None
        """
        if part_count is None:
            part_count = self.get_comm().get_size()
        file_list = get_part_file_list(file_prefix, part_count)
        self.from_filelist(file_list)

    def local_to_file(self, filename: str):
        """
        Store local tensor of WholeMemory Tensor to file, all ranks should
          call this together with different filename
        :param filename: file name of local tensor file.
        :return: None
        """
        self.wmb_tensor.to_file(filename)

    def to_file_prefix(self, file_prefix: str):
        """
        Store WholeMemory Tensor to files with same prefix.
        :param file_prefix: file name prefix
        :return: None
        """
        wm_comm = self.get_comm()
        filename = get_part_file_name(
            file_prefix, wm_comm.get_rank(), wm_comm.get_size()
        )
        self.local_to_file(filename)


def create_wholememory_tensor(
    comm: WholeMemoryCommunicator,
    memory_type: str,
    memory_location: str,
    sizes: List[int],
    dtype: "torch.dtype",
    strides: List[int],
    tensor_entry_partition: Union[List[int], None] = None,
):
    """
    Create empty WholeMemory Tensor. Now only support dim = 1 or 2
    :param comm: WholeMemoryCommunicator
    :param memory_type: WholeMemory type, should be continuous, chunked or distributed
    :param memory_location: WholeMemory location, should be cpu or cuda
    :param sizes: size of the tensor
    :param dtype: data type of the tensor
    :param strides: strides of the tensor
    :param tensor_entry_partition: rank partition based on entry;
      tensor_entry_partition[i] determines the entry count of rank
      i and shoud be a positive integer; the sum of tensor_entry_partition
      should equal to total entry count; entries will be equally partitioned if None
    :return: Allocated WholeMemoryTensor
    """
    dim = len(sizes)
    if dim < 1 or dim > 2:
        raise ValueError("Only dim 1 or 2 is supported now.")
    if strides is None:
        strides = [1] * dim
        strides[0] = sizes[1] if dim == 2 else 1
    else:
        assert len(strides) == dim
        assert strides[-1] == 1
        if dim == 2:
            assert strides[0] >= sizes[1]
    td = wmb.PyWholeMemoryTensorDescription()
    td.set_shape(sizes)
    td.set_stride(strides)
    td.set_dtype(torch_dtype_to_wholememory_dtype(dtype))

    wm_memory_type = str_to_wmb_wholememory_memory_type(memory_type)
    wm_location = str_to_wmb_wholememory_location(memory_location)

    return WholeMemoryTensor(
        wmb.create_wholememory_tensor(
            td, comm.wmb_comm, wm_memory_type, wm_location, tensor_entry_partition
        )
    )


def create_wholememory_tensor_from_filelist(
    comm: WholeMemoryCommunicator,
    memory_type: str,
    memory_location: str,
    filelist: Union[List[str], str],
    dtype: "torch.dtype",
    last_dim_size: int = 0,
    last_dim_strides: int = -1,
    tensor_entry_partition: Union[List[int], None] = None,
    file_format: str = "binary",
):
    """
    Create WholeMemory Tensor from a list of files.
    :param comm: WholeMemoryCommunicator
    :param memory_type: WholeMemory type, should be continuous, chunked or distributed
    :param memory_location: WholeMemory location, should be cpu or cuda
    :param filelist: list of files
    :param dtype: data type of the tensor
    :param last_dim_size: 0 for create 1-D array, positive value for
      create matrix column size
    :param last_dim_strides: stride of last_dim, -1 for same as size of last dim.
    :param tensor_entry_partition: rank partition based on entry;
      tensor_entry_partition[i] determines the entry count of rank
      i and shoud be a positive integer; the sum of tensor_entry_partition
      should equal to total entry count; entries will be equally partitioned if None
    :param file_format: file format, one of binary, pytorch, parquet, or auto
    :return: WholeMemoryTensor
    """
    filelist = _normalize_filelist(filelist)
    file_format = _resolve_file_format(filelist, file_format)
    if last_dim_strides == -1:
        last_dim_strides = last_dim_size if last_dim_size > 0 else 1
    total_entry_count = _get_filelist_entry_count(
        filelist, file_format, dtype, last_dim_size
    )
    if last_dim_size == 0:
        sizes = [total_entry_count]
        strides = [1]
    else:
        sizes = [total_entry_count, last_dim_size]
        strides = [last_dim_strides, 1]
    wm_tensor = create_wholememory_tensor(
        comm,
        memory_type,
        memory_location,
        sizes,
        dtype,
        strides,
        tensor_entry_partition,
    )
    wm_tensor.from_filelist(filelist, file_format=file_format)
    return wm_tensor


def destroy_wholememory_tensor(wm_tensor: WholeMemoryTensor):
    """
    Destroy allocated WholeMemory Tensor
    :param wm_tensor: WholeMemory Tensor
    :return: None
    """
    wmb.destroy_wholememory_tensor(wm_tensor.wmb_tensor)
    wm_tensor.wmb_tensor = None
