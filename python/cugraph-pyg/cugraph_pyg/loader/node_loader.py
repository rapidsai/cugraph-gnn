# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Union, Tuple, Callable, Optional

import cugraph_pyg
from cugraph_pyg.utils.imports import import_optional

from .utils import generate_seed

torch_geometric = import_optional("torch_geometric")
torch = import_optional("torch")


class NodeLoader:
    """
    Duck-typed version of torch_geometric.loader.NodeLoader.
    Loads samples from batches of input nodes using a
    `~cugraph_pyg.sampler.BaseSampler.sample_from_nodes`
    function.
    """

    def __init__(
        self,
        data: Union[
            "torch_geometric.data.Data",
            "torch_geometric.data.HeteroData",
            Tuple[
                "torch_geometric.data.FeatureStore", "torch_geometric.data.GraphStore"
            ],
        ],
        node_sampler: "cugraph_pyg.sampler.BaseSampler",
        input_nodes: "torch_geometric.typing.InputNodes" = None,
        input_time: "torch_geometric.typing.OptTensor" = None,
        transform: Optional[Callable] = None,
        transform_sampler_output: Optional[Callable] = None,
        filter_per_worker: Optional[bool] = None,
        custom_cls: Optional["torch_geometric.data.HeteroData"] = None,
        input_id: "torch_geometric.typing.OptTensor" = None,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        **kwargs,
    ):
        """
        Parameters
        ----------
            data: Data, HeteroData, or Tuple[FeatureStore, GraphStore]
                See torch_geometric.loader.NodeLoader.
            node_sampler: BaseSampler
                See torch_geometric.loader.NodeLoader.
            input_nodes: InputNodes
                See torch_geometric.loader.NodeLoader.
            input_time: OptTensor
                See torch_geometric.loader.NodeLoader.
            transform: Callable (optional, default=None)
                This argument currently has no effect.
            transform_sampler_output: Callable (optional, default=None)
                This argument currently has no effect.
            filter_per_worker: bool (optional, default=False)
                This argument currently has no effect.
            custom_cls: HeteroData
                This argument currently has no effect.  This loader will
                always return a Data or HeteroData object.
            input_id: OptTensor
                See torch_geometric.loader.NodeLoader.

        """
        if not isinstance(data, (list, tuple)) or not isinstance(
            data[1],
            (cugraph_pyg.data.graph_store.GraphStore,),
        ):
            # Will eventually automatically convert these objects to cuGraph objects.
            raise NotImplementedError("Currently can't accept non-cugraph graphs")

        if not isinstance(node_sampler, cugraph_pyg.sampler.BaseSampler):
            raise NotImplementedError("Must provide a cuGraph sampler")

        if filter_per_worker:
            warnings.warn("filter_per_worker is currently ignored")

        if custom_cls is not None:
            warnings.warn("custom_cls is currently ignored")

        if transform is not None:
            warnings.warn("transform is currently ignored.")

        if transform_sampler_output is not None:
            warnings.warn("transform_sampler_output is currently ignored.")

        (
            input_type,
            input_nodes,
            input_id,
        ) = torch_geometric.loader.utils.get_input_nodes(
            data,
            input_nodes,
            input_id,
        )
        input_nodes = input_nodes.detach().clone()

        if input_nodes.numel() < batch_size and drop_last:
            raise ValueError(
                "The number of input nodes is less than the batch size"
                " and drop_last is True. This will result in all batches"
                " being dropped. Either set drop_last to False or increase"
                " the number of nodes in input_nodes."
            )

        if input_type is not None:
            input_nodes += data[1]._vertex_offsets[input_type]

        self.__input_data = torch_geometric.sampler.NodeSamplerInput(
            input_id=torch.arange(len(input_nodes), dtype=torch.int64, device="cuda")
            if input_id is None
            else input_id,
            node=input_nodes,
            time=input_time,
            input_type=input_type,
        )

        self.__data = data

        self.__node_sampler = node_sampler

        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.__drop_last = drop_last

    def __iter__(self):
        if self.__shuffle:
            perm = torch.randperm(self.__input_data.node.numel())
        else:
            perm = torch.arange(self.__input_data.node.numel())

        if self.__drop_last:
            d = perm.numel() % self.__batch_size
            if d > 0:
                perm = perm[:-d]

        input_data = torch_geometric.sampler.NodeSamplerInput(
            input_id=self.__input_data.input_id[perm],
            node=self.__input_data.node[perm],
            time=None
            if self.__input_data.time is None
            else self.__input_data.time[perm],
            input_type=self.__input_data.input_type,
        )

        return cugraph_pyg.sampler.SampleIterator(
            self.__data,
            self.__node_sampler.sample_from_nodes(
                input_data, random_state=generate_seed()
            ),
        )
