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

import warnings
import os

from typing import Optional, Tuple, List

from cugraph_pyg.tensor import DistEmbedding, DistTensor
from cugraph_pyg.tensor.utils import has_nvlink_network, is_empty

from cugraph.utilities.utils import import_optional, MissingModule

torch = import_optional("torch")
torch_geometric = import_optional("torch_geometric")
tensordict = import_optional("tensordict")
wgth = import_optional("pylibwholegraph.torch")


class TensorDictFeatureStore(
    object
    if isinstance(torch_geometric, MissingModule)
    else torch_geometric.data.FeatureStore
):
    """
    A basic implementation of the PyG FeatureStore interface that stores
    feature data in a single TensorDict.  This type of feature store is
    not distributed, so each node will have to load the entire graph's
    features into memory.
    """

    def __init__(self):
        """
        Constructs an empty TensorDictFeatureStore.
        """
        super().__init__()

        self.__features = {}

    def _put_tensor(
        self,
        tensor: "torch_geometric.typing.FeatureTensorType",
        attr: "torch_geometric.data.feature_store.TensorAttr",
    ) -> bool:
        if attr.group_name in self.__features:
            td = self.__features[attr.group_name]
            batch_size = td.batch_size[0]

            if attr.is_set("index"):
                if attr.attr_name in td.keys():
                    if attr.index.shape[0] != batch_size:
                        raise ValueError(
                            "Leading size of index tensor "
                            "does not match existing tensors for group name "
                            f"{attr.group_name}; Expected {batch_size}, "
                            f"got {attr.index.shape[0]}"
                        )
                    td[attr.attr_name][attr.index] = tensor
                    return True
                else:
                    warnings.warn(
                        "Ignoring index parameter "
                        f"(attribute does not exist for group {attr.group_name})"
                    )

            if tensor.shape[0] != batch_size:
                raise ValueError(
                    "Leading size of input tensor does not match "
                    f"existing tensors for group name {attr.group_name};"
                    f" Expected {batch_size}, got {tensor.shape[0]}"
                )
        else:
            batch_size = tensor.shape[0]
            self.__features[attr.group_name] = tensordict.TensorDict(
                {}, batch_size=batch_size
            )

        self.__features[attr.group_name][attr.attr_name] = tensor
        return True

    def _get_tensor(
        self, attr: "torch_geometric.data.feature_store.TensorAttr"
    ) -> Optional["torch_geometric.typing.FeatureTensorType"]:
        if attr.group_name not in self.__features:
            return None

        if attr.attr_name not in self.__features[attr.group_name].keys():
            return None

        tensor = self.__features[attr.group_name][attr.attr_name]
        return (
            tensor
            if (attr.index is None or (not attr.is_set("index")))
            else tensor[attr.index]
        )

    def _remove_tensor(
        self, attr: "torch_geometric.data.feature_store.TensorAttr"
    ) -> bool:
        if attr.group_name not in self.__features:
            return False

        if attr.attr_name not in self.__features[attr.group_name].keys():
            return False

        del self.__features[attr.group_name][attr.attr_name]
        return True

    def _get_tensor_size(
        self, attr: "torch_geometric.data.feature_store.TensorAttr"
    ) -> Tuple:
        return self._get_tensor(attr).size()

    def get_all_tensor_attrs(
        self,
    ) -> List["torch_geometric.data.feature_store.TensorAttr"]:
        attrs = []
        for group_name, td in self.__features.items():
            for attr_name in td.keys():
                attrs.append(
                    torch_geometric.data.feature_store.TensorAttr(
                        group_name,
                        attr_name,
                    )
                )

        return attrs


class FeatureStore(
    object
    if isinstance(torch_geometric, MissingModule)
    else torch_geometric.data.FeatureStore
):
    """
    A basic implementation of the PyG FeatureStore interface that stores
    feature data in WholeGraph WholeMemory.  This type of feature store is
    distributed, and avoids data replication across workers.

    Data should be sliced before being passed into this feature store.
    That means each worker should have its own partition and put_tensor
    should be called for each worker's local partition.  When calling
    get_tensor, multi_get_tensor, etc., the entire tensor can be accessed
    regardless of what worker's partition the desired slice of the tensor
    is on.
    """

    def __init__(self, memory_type=None, location="cpu"):
        """
        Constructs an empty WholeFeatureStore.

        Parameters
        ----------
        memory_type: str (optional, default=None)
            Has no effect.  Retained for compatibility purposes.

        location: str(optional, default='cpu')
            The location ('cpu' or 'cuda') where data is stored.
        """
        super().__init__()

        self.__features = {}

        self.__wg_location = location

        if int(os.environ["LOCAL_WORLD_SIZE"]) == torch.distributed.get_world_size():
            self.__backend = "vmm"
        else:
            self.__backend = "vmm" if has_nvlink_network() else "nccl"

        if memory_type is not None:
            warnings.warn(
                "The memory_type argument is deprecated. "
                "Memory type is now automatically inferred."
            )

    def __make_wg_tensor(self, tensor, ix=None):
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        d = torch.tensor(tensor.dim(), device="cuda", dtype=torch.int64)

        global_d = torch.empty((world_size,), device="cuda", dtype=torch.int64)
        torch.distributed.all_gather_into_tensor(global_d, d)
        if not (global_d[0] == global_d).all():
            raise ValueError("Tensor dimension must be the same across ranks")

        ld = torch.tensor(
            0 if is_empty(tensor) else tensor.shape[0], device="cuda", dtype=torch.int64
        )
        sizes = torch.empty((world_size,), device="cuda", dtype=torch.int64)
        torch.distributed.all_gather_into_tensor(sizes, ld)

        dtypes = {}
        dtype_ids = {}
        for k, v in [
            (torch.float32, 0),
            (torch.float64, 1),
            (torch.int32, 2),
            (torch.int64, 3),
            (torch.bool, 4),
        ]:
            dtypes[k] = v
            dtype_ids[v] = k

        def _encode_dtype(dtype: torch.dtype):
            if dtype not in dtypes:
                raise ValueError(f"Unsupported dtype: {dtype}")
            return dtypes[dtype]

        def _decode_dtype(dtype_id: int):
            if dtype_id not in dtype_ids:
                raise ValueError(f"Unsupported dtype id: {dtype_id}")
            return dtype_ids[dtype_id]

        dtype = torch.tensor(
            _encode_dtype(tensor.dtype), device="cuda", dtype=torch.int64
        )
        global_dtype = torch.empty((world_size,), device="cuda", dtype=torch.int64)
        torch.distributed.all_gather_into_tensor(global_dtype, dtype)
        global_dtype = global_dtype[sizes > 0]
        if len(global_dtype) == 0:
            raise ValueError("Tensor is empty")
        if (global_dtype[0] == global_dtype).all():
            dtype = _decode_dtype(int(global_dtype[0]))
        else:
            raise ValueError("Tensor dtype must be the same across ranks")

        if tensor.dim() == 1:
            global_shape = [
                sizes.sum(),
            ]

            tx = DistTensor(
                None,
                shape=global_shape,
                dtype=dtype,
                device=self.__wg_location,
                backend=self.__backend,
            )
        elif tensor.dim() == 2:
            td = torch.tensor(
                -1 if is_empty(tensor) else tensor.shape[1],
                device="cuda",
                dtype=torch.int64,
            )
            global_td = torch.empty((world_size,), device="cuda", dtype=torch.int64)
            torch.distributed.all_gather_into_tensor(global_td, td)
            global_td = global_td[global_td > 0]

            if len(global_td) == 0:
                raise ValueError("Tensor is empty")

            if (global_td[0] == global_td).all():
                td = int(global_td[0])
            else:
                raise ValueError("Trailing dimensions must be the same across ranks")

            global_shape = [int(sizes.sum()), td]
            tx = DistEmbedding(
                None,
                shape=global_shape,
                dtype=dtype,
                device=self.__wg_location,
                backend=self.__backend,
            )

            if is_empty(tensor):
                tensor = tensor.reshape((-1, td))
        else:
            raise ValueError("Tensor must be 1D or 2D.")

        if ix is None:
            offset = sizes[:rank].sum() if rank > 0 else 0
            ix = torch.arange(
                offset, offset + tensor.shape[0], dtype=torch.int64, device="cuda"
            ).contiguous()

        if tensor.shape[0] != ix.shape[0]:
            raise ValueError("Shape mismatch")
        if ix.dim() != 1:
            raise ValueError("Index must be 1D")

        tx[ix] = tensor.cpu().clone(memory_format=torch.contiguous_format).pin_memory()
        return tx

    def _put_tensor(
        self,
        tensor: "torch_geometric.typing.FeatureTensorType",
        attr: "torch_geometric.data.feature_store.TensorAttr",
    ) -> bool:
        if attr.is_set("index") and attr.index is not None:
            if (attr.group_name, attr.attr_name) not in self.__features:
                self.__features[
                    (attr.group_name, attr.attr_name)
                ] = self.__make_wg_tensor(tensor, ix=attr.index)
        else:
            self.__features[(attr.group_name, attr.attr_name)] = self.__make_wg_tensor(
                tensor
            )

        return True

    def _get_tensor(
        self, attr: "torch_geometric.data.feature_store.TensorAttr"
    ) -> Optional["torch_geometric.typing.FeatureTensorType"]:
        if (attr.group_name, attr.attr_name) not in self.__features:
            return None

        emb = self.__features[attr.group_name, attr.attr_name]

        if attr.is_set("index") and attr.index is not None:
            return emb[attr.index]

        return emb

    def _remove_tensor(
        self, attr: "torch_geometric.data.feature_store.TensorAttr"
    ) -> bool:
        if (attr.group_name, attr.attr_name) not in self.__features:
            return False

        del self.__features[attr.group_name, attr.attr_name]
        return True

    def _get_tensor_size(
        self, attr: "torch_geometric.data.feature_store.TensorAttr"
    ) -> Tuple:
        return self.__features[attr.group_name, attr.attr_name].shape

    def get_all_tensor_attrs(
        self,
    ) -> List["torch_geometric.data.feature_store.TensorAttr"]:
        attrs = []
        for (group_name, attr_name) in self.__features.keys():
            attrs.append(
                torch_geometric.data.feature_store.TensorAttr(
                    group_name=group_name,
                    attr_name=attr_name,
                )
            )

        return attrs
