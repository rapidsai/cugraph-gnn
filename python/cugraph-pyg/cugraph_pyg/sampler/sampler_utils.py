# Copyright (c) 2022-2025, NVIDIA CORPORATION.
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


from typing import Tuple, Optional, Dict, Union

from math import ceil

from cugraph_pyg.data import GraphStore

from cugraph_pyg.utils.imports import import_optional
import cupy
import pylibcugraph

torch_geometric = import_optional("torch_geometric")

torch = import_optional("torch")
HeteroSamplerOutput = torch_geometric.sampler.base.HeteroSamplerOutput


def verify_metadata(metadata: Optional[Dict[str, Union[str, Tuple[str, str, str]]]]):
    if metadata is not None:
        for k, v in metadata.items():
            assert isinstance(k, str), "Metadata keys must be strings."
            if isinstance(v, tuple):
                assert len(v) == 3, "Metadata tuples must be of length 3."
                assert isinstance(
                    v[0], str
                ), "Metadata tuple must be of type (str, str, str)."
                assert isinstance(
                    v[1], str
                ), "Metadata tuple must be of type (str, str, str)."
                assert isinstance(
                    v[2], str
                ), "Metadata tuple must be of type (str, str, str)."
            else:
                assert isinstance(
                    v, str
                ), "Metadata values must be strings or tuples of strings."


def filter_cugraph_pyg_store(
    feature_store,
    graph_store,
    node,
    row,
    col,
    edge,
    clx,
) -> "torch_geometric.data.Data":
    data = torch_geometric.data.Data()

    data.edge_index = torch.stack([row, col], dim=0)

    required_attrs = []
    for attr in feature_store.get_all_tensor_attrs():
        attr.index = edge if isinstance(attr.group_name, tuple) else node
        required_attrs.append(attr)
        data.num_nodes = attr.index.size(0)

    tensors = feature_store.multi_get_tensor(required_attrs)
    for i, attr in enumerate(required_attrs):
        data[attr.attr_name] = tensors[i]

    return data


def neg_sample(
    graph_store: GraphStore,
    seed_src: "torch.Tensor",
    seed_dst: "torch.Tensor",
    batch_size: int,
    neg_sampling: "torch_geometric.sampler.NegativeSampling",
    time: "torch.Tensor",
    node_time: "torch.Tensor",
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    try:
        # Compatibility for PyG 2.5
        src_weight = neg_sampling.src_weight
        dst_weight = neg_sampling.dst_weight
    except AttributeError:
        src_weight = neg_sampling.weight
        dst_weight = neg_sampling.weight
    unweighted = src_weight is None and dst_weight is None

    # Require at least one negative edge per batch
    num_neg = max(
        int(ceil(neg_sampling.amount * seed_src.numel())),
        int(ceil(seed_src.numel() / batch_size)),
    )

    if node_time is None:
        result_dict = pylibcugraph.negative_sampling(
            graph_store._resource_handle,
            graph_store._graph,
            num_neg,
            vertices=None
            if unweighted
            else cupy.arange(src_weight.numel(), dtype="int64"),
            src_bias=None if src_weight is None else cupy.asarray(src_weight),
            dst_bias=None if dst_weight is None else cupy.asarray(dst_weight),
            remove_duplicates=False,
            remove_false_negatives=False,
            exact_number_of_samples=True,
            do_expensive_check=False,
        )

        src_neg = torch.as_tensor(result_dict["sources"], device="cuda")[:num_neg]
        dst_neg = torch.as_tensor(result_dict["destinations"], device="cuda")[:num_neg]

        # TODO modifiy the C API so this condition is impossible
        if src_neg.numel() < num_neg:
            num_gen = num_neg - src_neg.numel()
            src_neg = torch.concat(
                [
                    src_neg,
                    torch.randint(
                        0, src_neg.max(), (num_gen,), device="cuda", dtype=torch.int64
                    ),
                ]
            )
            dst_neg = torch.concat(
                [
                    dst_neg,
                    torch.randint(
                        0, dst_neg.max(), (num_gen,), device="cuda", dtype=torch.int64
                    ),
                ]
            )
        return src_neg, dst_neg
    raise NotImplementedError(
        "Temporal negative sampling is currently unimplemented in cuGraph-PyG"
    )


def neg_cat(
    seed_pos: "torch.Tensor", seed_neg: "torch.Tensor", pos_batch_size: int
) -> Tuple["torch.Tensor", int]:
    num_seeds = seed_pos.numel()
    num_batches = int(ceil(num_seeds / pos_batch_size))
    neg_batch_size = int(ceil(seed_neg.numel() / num_batches))

    batch_pos_offsets = torch.full((num_batches,), pos_batch_size).cumsum(-1)[:-1]
    seed_pos_splits = torch.tensor_split(seed_pos, batch_pos_offsets)

    batch_neg_offsets = torch.full((num_batches,), neg_batch_size).cumsum(-1)[:-1]
    seed_neg_splits = torch.tensor_split(seed_neg, batch_neg_offsets)

    return (
        torch.concatenate(
            [torch.concatenate(s) for s in zip(seed_pos_splits, seed_neg_splits)]
        ),
        neg_batch_size,
    )
