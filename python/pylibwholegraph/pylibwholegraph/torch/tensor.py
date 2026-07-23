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

# Structured formats must be decoded before they can be copied into WholeMemory.
# Bound both the row count and approximate decoded bytes so this temporary
# staging memory remains independent of the total dataset size.
_STRUCTURED_BATCH_SIZE = 64 * 1024
_STRUCTURED_BATCH_SIZE_BYTES = 16 * 1024 * 1024

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
    if isinstance(filelist, (str, os.PathLike)):
        normalized_filelist = [os.fspath(filelist)]
    else:
        normalized_filelist = [os.fspath(filename) for filename in filelist]
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


def _open_pytorch_tensor(filename: str):
    # mmap=True maps tensor storages instead of eagerly copying them into RAM.
    # The mapping can be large in virtual address space, but pages become
    # resident only as the chunk iterator below accesses them.
    try:
        value = torch.load(filename, map_location="cpu", mmap=True, weights_only=True)
    except RuntimeError as error:
        raise ValueError(
            f"PyTorch file {filename!r} cannot be memory-mapped. Re-save it with "
            "the current torch.save format or convert it to binary before loading "
            "it into WholeMemory."
        ) from error
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


def _parquet_file_metadata(filename: str, dim: int, last_dim_size: int) -> int:
    # Parquet footers contain row counts and schemas, so allocation can be
    # planned without decoding any data pages.
    parquet_file = pq.ParquetFile(filename)
    schema = parquet_file.schema_arrow
    column_count = len(schema)
    if column_count == 0:
        raise ValueError(f"Parquet file {filename!r} contains no columns")
    if dim == 1 and column_count != 1:
        raise ValueError(
            f"File {filename!r} has {column_count} columns, expected a 1-D tensor"
        )
    if dim == 2 and column_count != last_dim_size:
        raise ValueError(
            f"File {filename!r} has {column_count} columns, expected shape "
            f"(N, {last_dim_size})"
        )

    try:
        column_dtypes = [np.dtype(field.type.to_pandas_dtype()) for field in schema]
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            f"Parquet file {filename!r} must contain only scalar numeric columns"
        ) from error
    if any(
        not (np.issubdtype(dtype, np.number) or np.issubdtype(dtype, np.bool_))
        for dtype in column_dtypes
    ):
        raise ValueError(
            f"Parquet file {filename!r} must contain only scalar numeric columns"
        )
    return parquet_file.metadata.num_rows


def _parquet_batch_to_tensor(batch, filename: str):
    # Parquet is columnar whereas WholeMemory tensors are dense row-major
    # tensors. This conversion necessarily allocates one temporary dense batch,
    # which is bounded by _STRUCTURED_BATCH_SIZE_BYTES.
    try:
        columns = [column.to_numpy(zero_copy_only=False) for column in batch.columns]
        return torch.as_tensor(np.column_stack(columns))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Parquet file {filename!r} must contain only scalar numeric columns"
        ) from error


def _get_structured_batch_size(input_row_size: int, output_row_size: int) -> int:
    # Budget for decoded input, a dense row-major conversion, and a possible
    # dtype-converted output. PyTorch does not always need all three buffers,
    # but using the same conservative bound keeps both paths predictable.
    approximate_staging_row_size = 2 * input_row_size + output_row_size
    return min(
        _STRUCTURED_BATCH_SIZE,
        max(
            1,
            _STRUCTURED_BATCH_SIZE_BYTES // approximate_staging_row_size,
        ),
    )


def _validate_row_range(
    filename: str,
    file_format: str,
    row_count: int,
    row_start: int,
    row_end: Union[int, None],
) -> int:
    if row_end is None:
        row_end = row_count
    if row_start < 0 or row_end < row_start or row_end > row_count:
        raise ValueError(
            f"Invalid {file_format} row range [{row_start}, {row_end}) for "
            f"{row_count} rows in {filename!r}"
        )
    return row_end


def _iter_parquet_tensors(
    filename: str,
    dtype: "torch.dtype",
    dim: int,
    last_dim_size: int,
    row_start: int = 0,
    row_end: Union[int, None] = None,
):
    parquet_file = pq.ParquetFile(filename)
    row_count = _parquet_file_metadata(filename, dim, last_dim_size)
    source_row_size = sum(
        np.dtype(field.type.to_pandas_dtype()).itemsize
        for field in parquet_file.schema_arrow
    )
    output_row_size = torch.tensor([], dtype=dtype).element_size() * (
        last_dim_size if dim == 2 else 1
    )
    batch_size = _get_structured_batch_size(
        source_row_size,
        output_row_size,
    )
    row_end = _validate_row_range(filename, "Parquet", row_count, row_start, row_end)

    # Each rank scans metadata for every file but decodes only row groups that
    # intersect its local WholeMemory partition. At most one bounded batch is
    # live at a time.
    row_group_start = 0
    for row_group in range(parquet_file.num_row_groups):
        row_group_count = parquet_file.metadata.row_group(row_group).num_rows
        row_group_end = row_group_start + row_group_count
        if row_group_end <= row_start:
            row_group_start = row_group_end
            continue
        if row_group_start >= row_end:
            break

        batch_start = row_group_start
        for batch in parquet_file.iter_batches(
            batch_size=batch_size, row_groups=[row_group]
        ):
            batch_end = batch_start + batch.num_rows
            copy_start = max(row_start, batch_start)
            copy_end = min(row_end, batch_end)
            if copy_start < copy_end:
                tensor = _parquet_batch_to_tensor(batch, filename)
                tensor = tensor[copy_start - batch_start : copy_end - batch_start]
                if dim == 1:
                    tensor = tensor[:, 0]
                yield tensor.to(device="cpu", dtype=dtype).contiguous()
            batch_start = batch_end
        row_group_start = row_group_end


def _iter_pytorch_tensors(
    filename: str,
    dtype: "torch.dtype",
    dim: int,
    last_dim_size: int,
    row_start: int = 0,
    row_end: Union[int, None] = None,
):
    tensor = _open_pytorch_tensor(filename)
    _validate_tensor_shape(tensor, filename, dim, last_dim_size)
    row_end = _validate_row_range(
        filename, "PyTorch", tensor.shape[0], row_start, row_end
    )

    source_width = tensor.shape[1] if tensor.dim() == 2 else 1
    source_row_size = tensor.element_size() * source_width
    output_width = last_dim_size if dim == 2 else 1
    output_row_size = torch.tensor([], dtype=dtype).element_size() * output_width
    batch_size = _get_structured_batch_size(
        source_row_size,
        output_row_size,
    )

    # Slicing an mmap-backed tensor produces a view and faults in only the
    # pages needed by this batch. Dtype conversion or contiguity fixes are also
    # limited to this batch instead of the rank's entire partition.
    for batch_start in range(row_start, row_end, batch_size):
        batch_end = min(batch_start + batch_size, row_end)
        batch = tensor[batch_start:batch_end]
        if dim == 1 and batch.dim() == 2:
            batch = batch[:, 0]
        yield batch.to(device="cpu", dtype=dtype).contiguous()


def _iter_structured_tensors(
    filename: str,
    file_format: str,
    dtype: "torch.dtype",
    dim: int,
    last_dim_size: int,
    row_start: int = 0,
    row_end: Union[int, None] = None,
):
    if file_format == "parquet":
        yield from _iter_parquet_tensors(
            filename, dtype, dim, last_dim_size, row_start, row_end
        )
    elif file_format == "pytorch":
        yield from _iter_pytorch_tensors(
            filename, dtype, dim, last_dim_size, row_start, row_end
        )
    else:
        raise ValueError(f"Unsupported structured file format {file_format!r}")


def _validate_tensor_shape(tensor, filename: str, dim: int, last_dim_size: int):
    if dim == 1:
        if tensor.dim() == 2 and tensor.shape[1] == 1:
            return
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


def _get_file_entry_count(
    filename: str,
    file_format: str,
    dtype: "torch.dtype",
    last_dim_size: int,
) -> int:
    if file_format == "binary":
        element_size = torch.tensor([], dtype=dtype).element_size()
        file_entry_size = (
            element_size * last_dim_size if last_dim_size > 0 else element_size
        )
        file_size = get_file_size(filename)
        if file_size % file_entry_size != 0:
            raise ValueError(
                "File %s size is %d not multiple of %d"
                % (filename, file_size, file_entry_size)
            )
        return file_size // file_entry_size

    dim = 2 if last_dim_size > 0 else 1
    if file_format == "parquet":
        return _parquet_file_metadata(filename, dim, last_dim_size)
    # Loading a PyTorch checkpoint's object graph is required to discover its
    # shape, but mmap keeps the underlying tensor storage lazy and resident
    # memory bounded.
    tensor = _open_pytorch_tensor(filename)
    _validate_tensor_shape(tensor, filename, dim, last_dim_size)
    return tensor.shape[0]


def _get_filelist_entry_count(
    filelist: List[str],
    file_format: str,
    dtype: "torch.dtype",
    last_dim_size: int,
) -> int:
    # This is a metadata-only planning pass for Parquet and binary files and an
    # mmap-backed shape inspection for PyTorch files. No source tensors are
    # concatenated or materialized here.
    return sum(
        _get_file_entry_count(filename, file_format, dtype, last_dim_size)
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
          tensor saved in an mmap-capable format. Parquet files must contain
          only scalar numeric columns.
        :return: None
        """
        filelist = _normalize_filelist(filelist)
        file_format = _resolve_file_format(filelist, file_format)
        if file_format == "binary":
            # Raw binary already matches WholeMemory's dense layout, so retain
            # the native loader as the zero-decode, lowest-overhead fast path.
            self.wmb_tensor.from_filelist(filelist, round_robin_size)
            return

        if round_robin_size != 0:
            raise ValueError(
                "round_robin_size is only supported for binary file loading"
            )

        memory_location = self.wmb_tensor.get_wholememory_handle().get_memory_location()
        is_host_memory = memory_location == WholeMemoryMemoryLocation.MlHost

        # Request a view in the destination's native location. In particular,
        # CPU WholeMemory should be populated directly from the CPU staging
        # batches instead of mapping it through a CUDA view and performing an
        # unnecessary host-to-device copy.
        local_tensor, local_start = self.get_local_tensor(host_view=is_host_memory)
        local_end = local_start + local_tensor.shape[0]
        file_start = 0
        last_dim_size = self.shape[1] if self.dim() == 2 else 0

        # Treat the file list as one logical row-major tensor. Every rank maps
        # its local WholeMemory row interval onto that logical tensor and reads
        # only intersecting source rows.
        for filename in filelist:
            file_entry_count = _get_file_entry_count(
                filename, file_format, self.dtype, last_dim_size
            )
            file_end = file_start + file_entry_count
            copy_start = max(file_start, local_start)
            copy_end = min(file_end, local_end)
            if copy_start < copy_end:
                source_start = copy_start - file_start
                destination_start = copy_start - local_start
                source_end = copy_end - file_start
                for file_tensor in _iter_structured_tensors(
                    filename,
                    file_format,
                    self.dtype,
                    self.dim(),
                    last_dim_size,
                    source_start,
                    source_end,
                ):
                    copy_count = file_tensor.shape[0]
                    local_tensor[
                        destination_start : destination_start + copy_count
                    ].copy_(file_tensor)
                    destination_start += copy_count
            file_start = file_end

        if file_start != self.shape[0]:
            raise ValueError(
                f"Files contain {file_start} entries, but the WholeMemory tensor "
                f"expects {self.shape[0]} entries"
            )
        # CPU copies are synchronous. CUDA copies may be enqueued on the
        # current stream and must complete before ranks leave the load phase.
        if not is_host_memory:
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
    expected_entry_count: Union[int, None] = None,
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
    :param expected_entry_count: optional expected number of rows. An error is
      raised before allocation when the files contain a different row count.
    :return: WholeMemoryTensor
    """
    filelist = _normalize_filelist(filelist)
    file_format = _resolve_file_format(filelist, file_format)
    if last_dim_strides == -1:
        last_dim_strides = last_dim_size if last_dim_size > 0 else 1

    # Phase 1 only inspects file sizes/metadata (or mmap-backed PyTorch
    # metadata). The complete source dataset is never assembled in CPU memory.
    total_entry_count = _get_filelist_entry_count(
        filelist, file_format, dtype, last_dim_size
    )
    if expected_entry_count is not None and total_entry_count != expected_entry_count:
        raise ValueError(
            f"Files contain {total_entry_count} entries, but "
            f"expected_entry_count is {expected_entry_count}"
        )
    if last_dim_size == 0:
        sizes = [total_entry_count]
        strides = [1]
    else:
        sizes = [total_entry_count, last_dim_size]
        strides = [last_dim_strides, 1]

    # Phase 2 allocates the final destination before decoding any structured
    # data. from_filelist() then fills the local partition one bounded batch at
    # a time, keeping peak staging memory independent of dataset size.
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
