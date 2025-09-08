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

from typing import Optional, Union, Tuple, List, Literal

from cugraph_pyg.utils.imports import import_optional
from cugraph_pyg.tensor import DistTensor

torch = import_optional("torch")


class DistMatrix:
    """
    WholeGraph-backed Distributed Matrix Interface for PyTorch.
    """

    def __init__(
        self,
        src: Optional[
            Union[
                Tuple[torch.Tensor, torch.Tensor],
                Tuple[DistTensor, DistTensor],
                str,
                List[str],
            ]
        ] = None,
        shape: Optional[Union[list, tuple]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Literal["cpu", "cuda"]] = "cpu",
        backend: Optional[Literal["nccl", "vmm"]] = "nccl",
        format: Optional[Literal["csc", "coo"]] = "coo",
    ):
        self.__backend = backend
        self._format = format

        if isinstance(src, (tuple, list)):
            if isinstance(src[0], str):
                raise NotImplementedError(
                    "Constructing from a file or list of files is not yet supported."
                )
            else:
                if len(src) != 2:
                    raise ValueError("src must be a tuple of two tensors")
                self._col = DistTensor(
                    src=src[0], device=device, dtype=(dtype or src[0].dtype)
                )
                self._row = DistTensor(
                    src=src[1], device=device, dtype=(dtype or src[1].dtype)
                )

                if self._format == "coo":
                    if self._col.shape[0] != self._row.shape[0]:
                        raise ValueError(
                            "col and row must have the same number of "
                            "elements for COO format"
                        )
        elif src is None:
            if dtype is None or shape is None:
                raise ValueError("dtype and shape must be provided if src is None")
            if self._format != "coo":
                raise ValueError("Only COO format is supported for empty matrices")
            self._col = DistTensor(
                src=None,
                device=device,
                dtype=dtype,
                shape=(shape[0],),
                backend=self.__backend,
            )
            self._row = DistTensor(
                src=None,
                device=device,
                dtype=dtype,
                shape=(shape[1],),
                backend=self.__backend,
            )
        elif isinstance(src, str):
            raise NotImplementedError(
                "Constructing from a file or list of files is not yet supported."
            )
        else:
            raise ValueError("Invalid src type")

    def __setitem__(
        self,
        idx: Union[torch.Tensor, slice],
        val: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    ):
        if isinstance(idx, slice):
            size = self._col.shape[0]
            idx = torch.arange(size)[idx]

        if self._format != "coo":
            raise ValueError("Updating is currently only supported for COO format")
        if isinstance(val, torch.Tensor):
            if val.dim() != 2:
                raise ValueError("val must be a 2D tensor")
            if val.shape[0] != 2:
                raise ValueError("val must be a 2xN tensor")
            if val.shape[1] != idx.shape[0]:
                raise ValueError("val and idx must have compatible shapes")
            self._col[idx] = val[0]
            self._row[idx] = val[1]
        elif isinstance(val, tuple):
            if len(val) != 2:
                raise ValueError("val must be a tuple of two tensors")
            self._col[idx] = val[0]
            self._row[idx] = val[1]

    def __getitem__(self, idx: torch.Tensor) -> torch.Tensor:
        if self._format != "coo":
            raise ValueError("Getting is currently only supported for COO format")
        if idx.dim() != 1:
            raise ValueError("idx must be a 1D tensor")

        return torch.stack([self._col[idx], self._row[idx]])

    def get_local_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (self._col.get_local_tensor(), self._row.get_local_tensor())

    @property
    def local_col(self) -> torch.Tensor:
        return self._col.get_local_tensor()

    @property
    def local_row(self) -> torch.Tensor:
        return self._row.get_local_tensor()

    @property
    def local_coo(self) -> torch.Tensor:
        return torch.stack(self.get_local_tensor())

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._col.shape[0], self._row.shape[0])

    @property
    def dtype(self) -> torch.dtype:
        return self._col.dtype
