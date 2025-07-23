# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

from typing import List, Optional, Union, Literal

import numpy as np

from cugraph_pyg.tensor.utils import (
    copy_host_global_tensor_to_local,
    create_wg_dist_tensor,
    create_wg_dist_tensor_from_files,
)

from cugraph.utilities.utils import import_optional

torch = import_optional("torch")
wgth = import_optional("pylibwholegraph.torch")
pylibwholegraph = import_optional("pylibwholegraph")


class DistTensor:
    """
    WholeGraph-backed Distributed Tensor Interface for PyTorch.
    Parameters
    ----------
    src: Optional[Union[torch.Tensor, str, List[str]]]
        The source of the tensor. It can be a torch.Tensor on host, a file path,
        or a list of file paths.
        When the source is omitted, the tensor will be load later.
    shape : Optional[list, tuple]
        The shape of the tensor. It has to be a one- or two-dimensional tensor
        for now.
        When the shape is omitted, the `src` has to be specified and must
        be `pt` or `npy` file paths.
    dtype : Optional[torch.dtype]
        The dtype of the tensor.
        When the dtype is omitted, the `src` has to be specified
        and must be `pt` or `npy` file paths.
    device : Optional[Literal["cpu", "cuda"]] = "cpu"
        The desired location to store the embedding [ "cpu" | "cuda" ].
        Default is "cpu", i.e., host-pinned memory (UVA).
    partition_book : Union[List[int], None] = None
        1-D Range partition based on entry (dim-0).
        partition_book[i] determines the
        entry count of rank i and shoud be a positive integer;
        the sum of partition_book should equal to shape[0].
        Entries will be equally partitioned if None.
    backend : Optional[Literal["vmm", "nccl", "nvshmem", "chunked"]] = "nccl"
        The backend used for communication. Default is "nccl".
    """

    def __init__(
        self,
        src: Optional[Union["torch.Tensor", str, List[str]]] = None,
        shape: Optional[Union[list, tuple]] = None,
        dtype: Optional["torch.dtype"] = None,
        device: Optional[Literal["cpu", "cuda"]] = "cpu",
        partition_book: Optional[
            Union[List[int], None]
        ] = None,  # location memtype ?? backend?? ; engine; comm =  vmm/nccl ..
        backend: Optional[str] = "nccl",
        *args,
        **kwargs,
    ):

        self._tensor = None
        self.__device = device
        if src is None:
            # Create an empty WholeGraph tensor
            if shape is None:
                raise ValueError("Please specify the shape of the tensor.")
            elif dtype is None:
                raise ValueError("Please specify the dtype of the tensor.")
            elif not (len(shape) in [1, 2]):
                raise ValueError("The shape of the tensor must be 1D or 2D.")

            self._tensor = create_wg_dist_tensor(
                list(shape), dtype, device, partition_book, backend, *args, **kwargs
            )
            self.__dtype = dtype
        else:
            if isinstance(src, list):
                # A list of file paths for a tensor
                # Only support the binary file format directly loaded via WM API for now
                if shape is None or dtype is None:
                    raise ValueError(
                        "For now, reading from multiple files is only"
                        " supported with binary format."
                    )

                self._tensor = create_wg_dist_tensor_from_files(
                    src, shape, dtype, device, partition_book, backend, *args, **kwargs
                )
                self.__dtype = dtype
            else:
                self._init_from_single_source(
                    src, device, partition_book, backend, *args, **kwargs
                )

    def _init_from_single_source(
        self, src, device, partition_book, backend, *args, **kwargs
    ):
        """
        Initialize DistTensor from a single source (tensor or file).

        Parameters
        ----------
        src : Union[torch.Tensor, str]
            Source tensor or file path
        device : str
            Device to store the tensor on
        partition_book : Union[List[int], None]
            Partition configuration
        backend : str
            Communication backend
        """
        if isinstance(src, torch.Tensor):
            self._tensor = create_wg_dist_tensor(
                list(src.shape),
                src.dtype,
                device,
                partition_book,
                backend,
                *args,
                **kwargs,
            )
            self.__dtype = src.dtype
            host_tensor = src
        elif isinstance(src, str) and src.endswith(".pt"):
            host_tensor = torch.load(src, mmap=True)
            self._tensor = create_wg_dist_tensor(
                list(host_tensor.shape),
                host_tensor.dtype,
                device,
                partition_book,
                backend,
                *args,
                **kwargs,
            )
            self.__dtype = host_tensor.dtype
        elif isinstance(src, str) and src.endswith(".npy"):
            host_tensor = torch.from_numpy(np.load(src, mmap_mode="c"))
            self.__dtype = host_tensor.dtype
            self._tensor = create_wg_dist_tensor(
                list(host_tensor.shape),
                host_tensor.dtype,
                device,
                partition_book,
                backend,
                *args,
                **kwargs,
            )
        else:
            raise ValueError(
                "Unsupported source type. Please provide "
                "a torch.Tensor, a file path, or a list of file paths."
            )

        self.load_from_global_tensor(host_tensor)

    def load_from_global_tensor(self, tensor):
        # input pytorch host tensor (mmapped or in shared host memory),
        # and copy to wholegraph tensor
        if self._tensor is None:
            raise ValueError("Please create WholeGraph tensor first.")

        self.__dtype = tensor.dtype
        if isinstance(self._tensor, wgth.WholeMemoryEmbedding):
            _tensor = self._tensor.get_embedding_tensor()
        else:
            _tensor = self._tensor
        copy_host_global_tensor_to_local(_tensor, tensor, _tensor.get_comm())

    def load_from_local_tensor(self, tensor):
        # input pytorch host tensor (mmapped or in shared host memory),
        # and copy to wholegraph tensor
        if self._tensor is None:
            raise ValueError("Please create WholeGraph tensor first.")

        if self._tensor.local_shape != tensor.shape:
            raise ValueError(
                "The shape of the tensor does not match the shape of the local tensor."
            )
        if self.dtype != tensor.dtype:
            raise ValueError(
                "The dtype of the tensor does not match the dtype of the local tensor."
            )

        if isinstance(self._tensor, wgth.WholeMemoryEmbedding):
            self._tensor.get_embedding_tensor().get_local_tensor().copy_(tensor)
        else:
            self._tensor.get_local_tensor().copy_(tensor)

    @classmethod
    def from_tensor(
        cls,
        tensor: "torch.Tensor",
        device: Optional[Literal["cpu", "cuda"]] = "cpu",
        partition_book: Union[List[int], None] = None,
        backend: Optional[str] = "nccl",
    ):
        """Create a WholeGraph-backed Distributed Tensor from a PyTorch tensor.
        Parameters
        ----------
        tensor : torch.Tensor
            The PyTorch tensor to be copied to the WholeGraph tensor.
        device : str, optional
            The desired location to store the embedding [ "cpu" | "cuda" ].
            Default is "cpu".
        backend : str, optional
            The backend used for communication. Default is "nccl".
        Returns:
        -------
        DistTensor
            The WholeGraph-backed Distributed Tensor.
        """
        return cls(
            src=tensor, device=device, partition_book=partition_book, backend=backend
        )

    @classmethod
    def from_file(
        cls,
        file_path: str,
        device: Optional[Literal["cpu", "cuda"]] = "cpu",
        partition_book: Union[List[int], None] = None,
        backend: Optional[str] = "nccl",
    ):
        """Create a WholeGraph-backed Distributed Tensor from a file.
        Parameters
        ----------
        file_path : str
            The file path to the tensor.
            The file can be in the format of PyTorch tensor or NumPy array.
        device : str, optional
            The desired location to store the embedding [ "cpu" | "cuda" ].
            Default is "cpu".
        backend : str, optional
            The backend used for communication. Default is "nccl".
        Returns:
        -------
        DistTensor
            The WholeGraph-backed Distributed Tensor.
        """
        return cls(
            src=file_path, device=device, partition_book=partition_book, backend=backend
        )

    def __setitem__(self, idx: "torch.Tensor", val: "torch.Tensor"):
        """Set the embeddings for the specified node indices.
        This call must be called by all processes.
        Parameters
        ----------
        idx : torch.Tensor
            Index of the embeddings to collect.
        val : torch.Tensor
            The requested node embeddings.
        """
        assert self._tensor is not None, "Please create WholeGraph tensor first."
        if not idx.is_cuda:
            idx = idx.pin_memory()
        if not val.is_cuda:
            val = val.pin_memory()

        if val.dtype != self.dtype:
            val = val.to(self.dtype)
        self._tensor.scatter(val, idx)

    def __getitem__(self, idx: "torch.Tensor") -> "torch.Tensor":
        """Get the embeddings for the specified node indices (remotely).
        This call must be called by all processes.
        Parameters
        ----------
        idx : torch.Tensor
            Index of the embeddings to collect.
        Returns:
        -------
        torch.Tensor
            The requested node embeddings.
        """
        assert self._tensor is not None, "Please create WholeGraph tensor first."
        if not idx.is_cuda:
            idx = idx.pin_memory()
        output_tensor = self._tensor.gather(idx)  # output_tensor is on cuda by default
        return output_tensor

    def get_local_tensor(self, host_view=False):
        """Get the local embedding tensor and its element offset at current rank.
        Returns:
        -------
        (torch.Tensor, int)
            Tuple of local torch Tensor (converted from DLPack) and its offset.
        """
        local_tensor, offset = self._tensor.get_local_tensor(host_view=host_view)
        return local_tensor

    def get_local_offset(self):
        """Get the local embedding tensor and its element offset at current rank.
        Returns:
        -------
        (torch.Tensor, int)
            Tuple of local torch Tensor (converted from DLPack) and its offset.
        """
        _, offset = self._tensor.get_local_tensor()
        return offset

    def get_comm(self):
        """Get the communicator of the WholeGraph embedding.
        Returns:
        -------
        WholeMemoryCommunicator
            The WholeGraph global communicator of the WholeGraph embedding.
        """
        assert self._tensor is not None, "Please create WholeGraph tensor first."
        return self._tensor.get_comm()

    @property
    def dim(self):
        return self._tensor.dim()

    @property
    def shape(self):
        return self._tensor.shape

    @property
    def device(self):
        return self.__device

    @property
    def dtype(self):
        return self.__dtype

    def __repr__(self):
        if self._tensor is None:
            return "<DistTensor: No tensor loaded>"

        # Format the output similar to PyTorch
        tensor_repr = "DistTensor("
        tensor_repr += (
            f"shape={self._tensor.shape}, dtype={self.dtype}, device='{self.device}')"
        )
        return tensor_repr


class DistEmbedding(DistTensor):
    """WholeGraph-backed Distributed Embedding Interface for PyTorch.
    Parameters
    ----------
    src: Optional[Union[torch.Tensor, str, List[str]]]
        The source of the tensor. It can be a torch.Tensor on host,
        a file path, or a list of file paths.
        When the source is omitted, the tensor will be load later.
    shape : Optional[list, tuple]
        The shape of the tensor. It has to be a one- or two-dimensional tensor
        for now.
        When the shape is omitted, the `src` has to be specified and
        must be `pt` or `npy` file paths.
    dtype : Optional[torch.dtype]
        The dtype of the tensor.
        Whne the dtype is omitted, the `src` has to be specified
        and must be `pt` or `npy` file paths.
    device : Optional[Literal["cpu", "cuda"]] = "cpu"
        The desired location to store the embedding [ "cpu" | "cuda" ].
        Default is "cpu", i.e., host-pinned memory (UVA).
    partition_book : Union[List[int], None] = None
        1-D Range partition based on entry (dim-0). partition_book[i] determines the
        entry count of rank i and shoud be a positive integer;
        the sum of partition_book should equal to shape[0].
        Entries will be equally partitioned if None.
    backend : Optional[Literal["vmm", "nccl", "nvshmem", "chunked"]] = "nccl"
        The backend used for communication. Default is "nccl".
    cache_policy : Optional[WholeMemoryCachePolicy] = None
        The cache policy for the tensor if it is an embedding. Default is None.
    gather_sms : Optional[int] = -1
        Whether to gather the embeddings on all GPUs. Default is False.
    round_robin_size: int = 0
        continuous embedding size of a rank using round robin shard strategy
    name : Optional[str]
        The name of the tensor.
    """

    def __init__(
        self,
        src: Optional[Union["torch.Tensor", str, List[str]]] = None,
        shape: Optional[Union[list, tuple]] = None,
        dtype: Optional["torch.dtype"] = None,
        device: Optional[Literal["cpu", "cuda"]] = "cpu",
        partition_book: Union[List[int], None] = None,
        backend: Optional[str] = "nccl",
        cache_policy: Optional["pylibwholegraph.WholeMemoryCachePolicy"] = None,
        gather_sms: Optional[int] = -1,
        round_robin_size: int = 0,
        name: Optional[str] = None,
    ):
        self._name = name

        super().__init__(
            src,
            shape,
            dtype,
            device,
            partition_book,
            backend,
            cache_policy=cache_policy,
            gather_sms=gather_sms,
            round_robin_size=round_robin_size,
        )
        self._embedding = self._tensor  # returned _tensor is a WmEmbedding object
        self._tensor = self._embedding.get_embedding_tensor()

    @classmethod
    def from_tensor(
        cls,
        tensor: "torch.Tensor",
        device: Literal["cpu", "cuda"] = "cpu",
        partition_book: Union[List[int], None] = None,
        name: Optional[str] = None,
        cache_policy=None,
        *args,
        **kwargs,
    ):
        """
        Create a WholeGraph-backed Distributed Embedding
        (hooked with PyT's grad tracing) from a PyTorch tensor.
        Parameters
        ----------
        tensor : torch.Tensor
            The PyTorch tensor to be copied to the WholeGraph tensor.
        device : str, optional
            The desired location to store the embedding [ "cpu" | "cuda" ].
            Default is "cpu".
        name : str, optional
            The name of the tensor.
        Returns:
        -------
        DistEmbedding
            The WholeGraph-backed Distributed Tensor.
        """
        return cls(
            src=tensor,
            device=device,
            partition_book=partition_book,
            name=name,
            cache_policy=cache_policy,
            *args,
            **kwargs,
        )

    @classmethod
    def from_file(
        cls,
        file_path: str,
        device: Literal["cpu", "cuda"] = "cpu",
        partition_book: Union[List[int], None] = None,
        name: Optional[str] = None,
        cache_policy=None,
        *args,
        **kwargs,
    ):
        """Create a WholeGraph-backed Distributed Tensor from a file.
        Parameters
        ----------
        file_path : str
            The file path to the tensor. The file can be in the
            format of PyTorch tensor or NumPy array.
        device : str, optional
            The desired location to store the embedding [ "cpu" | "cuda" ].
            Default is "cpu".
        name : str, optional
            The name of the tensor.
        Returns:
        -------
        DistTensor
            The WholeGraph-backed Distributed Tensor.
        """
        return cls(
            src=file_path,
            device=device,
            partition_book=partition_book,
            name=name,
            cache_policy=cache_policy,
            *args,
            **kwargs,
        )

    def __setitem__(self, idx: "torch.Tensor", val: "torch.Tensor"):
        """Set the embeddings for the specified node indices.
        This call must be called by all processes.
        Parameters
        ----------
        idx : torch.Tensor
            Index of the embeddings to collect.
        val : torch.Tensor
            The requested node embeddings.
        """
        assert self._tensor is not None, "Please create WholeGraph tensor first."
        if not idx.is_cuda:
            idx = idx.pin_memory()
        if not val.is_cuda:
            val = val.pin_memory()

        if val.dtype != self.dtype:
            val = val.to(self.dtype)
        self._embedding.get_embedding_tensor().scatter(val, idx)

    def __getitem__(self, idx: "torch.Tensor") -> "torch.Tensor":
        """Get the embeddings for the specified node indices (remotely).
        This call must be called by all processes.
        Parameters
        ----------
        idx : torch.Tensor
            Index of the embeddings to collect.
        Returns:
        -------
        torch.Tensor
            The requested node embeddings.
        """
        assert self._tensor is not None, "Please create WholeGraph tensor first."
        if not idx.is_cuda:
            idx = idx.pin_memory()
        output_tensor = self._embedding.gather(
            idx
        )  # output_tensor is on cuda by default
        return output_tensor

    @property
    def name(self):
        return self._name

    def __repr__(self):
        if self._embedding is None:
            return f"<DistEmbedding: No embedding loaded, Name: {self._name}>"

        # Format the output similar to PyTorch
        tensor_repr = "DistEmbedding("
        if self._name:
            tensor_repr += f"name={self._name}, "
        tensor_repr += (
            f"shape={self.shape}, dtype={self.dtype}, device='{self.device}')"
        )
        return tensor_repr
