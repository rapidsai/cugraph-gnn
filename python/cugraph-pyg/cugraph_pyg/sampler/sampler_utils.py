# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


from typing import Tuple, Optional, Dict, Union, Callable

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
                assert isinstance(v[0], str), (
                    "Metadata tuple must be of type (str, str, str)."
                )
                assert isinstance(v[1], str), (
                    "Metadata tuple must be of type (str, str, str)."
                )
                assert isinstance(v[2], str), (
                    "Metadata tuple must be of type (str, str, str)."
                )
            else:
                assert isinstance(v, str), (
                    "Metadata values must be strings or tuples of strings."
                )


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


def _call_plc_negative_sampling(
    graph_store,
    num_neg,
    vertices,
    src_weight,
    dst_weight,
    remove_duplicates=False,
    remove_false_negatives=False,
    exact_number_of_samples=False,
):
    result_dict = pylibcugraph.negative_sampling(
        graph_store._resource_handle,
        graph_store._graph,
        num_neg,
        vertices=None if vertices is None else cupy.asarray(vertices),
        src_bias=None if src_weight is None else cupy.asarray(src_weight),
        dst_bias=None if dst_weight is None else cupy.asarray(dst_weight),
        remove_duplicates=remove_duplicates,
        remove_false_negatives=remove_false_negatives,
        exact_number_of_samples=exact_number_of_samples,
        do_expensive_check=False,
    )
    src_neg = torch.as_tensor(result_dict["sources"], device="cuda")[:num_neg]
    dst_neg = torch.as_tensor(result_dict["destinations"], device="cuda")[:num_neg]
    return src_neg, dst_neg


def neg_sample(
    graph_store: GraphStore,
    seed_src: "torch.Tensor",
    seed_dst: "torch.Tensor",
    input_type: Tuple[str, str, str],
    batch_size: int,
    neg_sampling: "torch_geometric.sampler.NegativeSampling",
    seed_time: Optional["torch.Tensor"] = None,
    node_time_func: Callable[[str, "torch.Tensor"], "torch.Tensor"] = None,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    # TODO Add support for remove_duplicates, remove_false_negatives (rapidsai/cugraph-gnn#378)
    try:
        # Compatibility for PyG 2.5
        src_weight = neg_sampling.src_weight
        dst_weight = neg_sampling.dst_weight
    except AttributeError:
        src_weight = neg_sampling.weight
        dst_weight = neg_sampling.weight

    # Require at least one negative edge per batch
    num_neg = max(
        int(ceil(neg_sampling.amount * seed_src.numel())),
        int(ceil(seed_src.numel() / batch_size)),
    )

    # The weights need to match the expected number of nodes
    if graph_store.is_homogeneous:
        num_src_nodes = num_dst_nodes = list(graph_store._num_vertices().values())[0]
    else:
        num_src_nodes = graph_store._num_vertices()[input_type[0]]
        num_dst_nodes = graph_store._num_vertices()[input_type[2]]

    if src_weight is not None and dst_weight is not None:
        if src_weight.dtype != dst_weight.dtype:
            raise ValueError(
                f"The 'src_weight' and 'dst_weight' attributes need to have the same"
                f" dtype (got {src_weight.dtype} and {dst_weight.dtype})"
            )
    weight_dtype = (
        torch.float32
        if (src_weight is None and dst_weight is None)
        else (src_weight.dtype if src_weight is not None else dst_weight.dtype)
    )

    if src_weight is None:
        src_weight = torch.ones(num_src_nodes, dtype=weight_dtype, device="cuda")
    else:
        if src_weight.numel() != num_src_nodes:
            raise ValueError(
                f"The 'src_weight' attribute needs to match the number of source nodes"
                f" {num_src_nodes} (got {src_weight.numel()})"
            )

    if dst_weight is None:
        dst_weight = torch.ones(num_dst_nodes, dtype=weight_dtype, device="cuda")
    else:
        if dst_weight.numel() != num_dst_nodes:
            raise ValueError(
                f"The 'dst_weight' attribute needs to match the number of destination"
                f" nodes {num_dst_nodes} (got {dst_weight.numel()})"
            )

    # If the graph is heterogeneous, the weights need to be concatenated together
    # and offsetted.
    if not graph_store.is_homogeneous:
        if input_type[0] != input_type[2]:
            vertices = torch.concat(
                [
                    torch.arange(num_src_nodes, dtype=torch.int64, device="cuda")
                    + graph_store._vertex_offsets[input_type[0]],
                    torch.arange(num_dst_nodes, dtype=torch.int64, device="cuda")
                    + graph_store._vertex_offsets[input_type[2]],
                ]
            )
        else:
            vertices = (
                torch.arange(num_src_nodes, dtype=torch.int64, device="cuda")
                + graph_store._vertex_offsets[input_type[0]]
            )

        src_weight = torch.concat(
            [src_weight, torch.zeros(num_dst_nodes, dtype=weight_dtype, device="cuda")]
        )
        dst_weight = torch.concat(
            [torch.zeros(num_src_nodes, dtype=weight_dtype, device="cuda"), dst_weight]
        )
    elif src_weight is None and dst_weight is None:
        vertices = None
    else:
        vertices = torch.arange(num_src_nodes, dtype=torch.int64, device="cuda")

    src_neg, dst_neg = _call_plc_negative_sampling(
        graph_store, num_neg, vertices, src_weight, dst_weight
    )

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

    if node_time_func is not None:
        # Temporal negative sampling - node_time must be <= seed_time
        # Seed time is both src/dst time in the PyG API.
        # TODO maybe handle this in the C API?
        num_neg_per_pos = int(ceil(neg_sampling.amount))
        seed_time = seed_time.view(1, -1).expand(num_neg_per_pos, -1).flatten().cuda()

        # For homogeneous graphs, input_type is None, so get the single node type
        if graph_store.is_homogeneous:
            node_type = list(graph_store._vertex_offsets.keys())[0]
            node_offset = graph_store._vertex_offsets[node_type]
            src_node_type = dst_node_type = node_type
            src_node_offset = dst_node_offset = node_offset
        else:
            src_node_type = input_type[0]
            dst_node_type = input_type[2]
            src_node_offset = graph_store._vertex_offsets[src_node_type]
            dst_node_offset = graph_store._vertex_offsets[dst_node_type]

        src_node_time = node_time_func(src_node_type, src_neg - src_node_offset)
        dst_node_time = node_time_func(dst_node_type, dst_neg - dst_node_offset)

        target_samples = src_neg.numel()
        valid_mask = (src_node_time <= seed_time) & (dst_node_time <= seed_time)
        src_neg = src_neg[valid_mask]
        dst_neg = dst_neg[valid_mask]
        target_samples = src_neg.numel()
        seed_time = seed_time[~valid_mask]

        # Matches the PyG API, attempts 5 times.
        for _ in range(5):
            diff = target_samples - src_neg.numel()
            if diff <= 0:
                break
            src_neg_p, dst_neg_p = _call_plc_negative_sampling(
                graph_store, diff, vertices, src_weight, dst_weight
            )

            src_time_p = node_time_func(src_node_type, src_neg_p - src_node_offset)
            dst_time_p = node_time_func(dst_node_type, dst_neg_p - dst_node_offset)

            valid_mask = (src_time_p <= seed_time) & (dst_time_p <= seed_time)
            src_neg_p = src_neg_p[valid_mask]
            dst_neg_p = dst_neg_p[valid_mask]
            src_neg = torch.concat([src_neg, src_neg_p])
            dst_neg = torch.concat([dst_neg, dst_neg_p])
            seed_time = seed_time[~valid_mask]

        diff = target_samples - src_neg.numel()
        if diff > 0:
            # Select the earliest occuring node for src/dst
            # Again, this matches the PyG API.
            src_neg_p, dst_neg_p = _call_plc_negative_sampling(
                graph_store, diff, vertices, src_weight, dst_weight
            )

            src_time_p = node_time_func(src_node_type, src_neg_p - src_node_offset)
            invalid_src = src_time_p[src_time_p > seed_time]
            src_neg_p[invalid_src] = src_neg[
                node_time_func(src_node_type, src_neg - src_node_offset).argmin()
            ]

            dst_time_p = node_time_func(dst_node_type, dst_neg_p - dst_node_offset)
            invalid_dst = dst_time_p[dst_time_p > seed_time]
            dst_neg_p[invalid_dst] = dst_neg[
                node_time_func(dst_node_type, dst_neg - dst_node_offset).argmin()
            ]
            src_neg = torch.concat([src_neg, src_neg_p])
            dst_neg = torch.concat([dst_neg, dst_neg_p])

    # The returned negative edges already have offsetted vertex IDs,
    # and are valid input for the pylibcugraph sampler.
    return src_neg, dst_neg


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
