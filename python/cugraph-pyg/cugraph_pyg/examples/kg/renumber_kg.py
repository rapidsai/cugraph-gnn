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

# Execute this script with torchrun (see run_renumber.sh - Each process needs
# to call torchrun separately)

import os
import argparse

import pandas  # consider using cudf.pandas here

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--node_types",
        type=str,
        required=True,
        help="List of node types separated by commas (i.e. shape,size,length)",
    )
    parser.add_argument(
        "--node_input_folders",
        type=str,
        required=True,
        help=(
            "List of folders containing input node IDs (should match node type order)"
            " (i.e. data/shape, data/size, data/length)."
            " Each folder should contain (# local workers) files."
        ),
    )
    parser.add_argument(
        "--node_output_folders",
        type=str,
        required=True,
        help=(
            "List of folders containing output node IDs (should match node type order)"
            " (i.e. data/shape, data/size, data/length)."
        ),
    )
    parser.add_argument(
        "--node_colname",
        type=str,
        required=True,
        help="Name of the column containing node ids in each node file.",
    )
    parser.add_argument(
        "--edge_types",
        type=str,
        required=True,
        help=(
            "List of canonical edge types separated by semicolons"
            " (i.e. paper,cites,paper;author,writes,paper)"
        ),
    )
    parser.add_argument(
        "--edge_input_folders",
        type=str,
        required=True,
        help=(
            "List of input folders containing edges, separated by commas"
            " (i.e. data/paper_cites_paper,data/author_writes_paper). "
            "Each folder should contain (# local workers) files."
        ),
    )
    parser.add_argument(
        "--edge_output_folders",
        type=str,
        required=True,
        help=(
            "List of output folders containing edges, separated by commas "
            "(i.e. data/paper_cites_paper,data/author_writes_paper)."
        ),
    )
    parser.add_argument(
        "--source_colname",
        type=str,
        required=True,
        help="Name of the column in each edge file corresponding to source node id.",
    )
    parser.add_argument(
        "--destination_colname",
        type=str,
        required=True,
        help=(
            "Name of the column in each edge file corresponding to"
            " destination node id."
        ),
    )
    parser.add_argument(
        "--output_format",
        type=str,
        required=False,
        default="csv",
        help="csv or parquet",
    )
    parser.add_argument(
        "--input_format", type=str, required=False, default="csv", help="csv or parquet"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    torch.distributed.init_process_group("nccl")
    world_size = torch.distributed.get_world_size()
    global_rank = torch.distributed.get_rank()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(local_rank)
    torch.cuda.set_device(local_rank)

    node_types = args.node_types.split(",")
    local_num_nodes = {}
    global_num_nodes = {}
    local_node_offsets = {}
    global_renumber_map = {}

    for folder in args.node_output_folders.split(","):
        os.makedirs(folder, exist_ok=True)
    for folder in args.edge_output_folders.split(","):
        os.makedirs(folder, exist_ok=True)

    for node_type, node_folder_name, output_folder_name in zip(
        node_types,
        args.node_input_folders.split(","),
        args.node_output_folders.split(","),
    ):
        node_fname = sorted(os.listdir(node_folder_name))[local_rank]
        node_fpath = os.path.join(node_folder_name, node_fname)

        if args.input_format.lower() == "csv":
            ndf = pandas.read_csv(node_fpath)
        elif args.input_format.lower() == "parquet":
            ndf = pandas.read_parquet(node_fpath)
        else:
            raise ValueError("Invalid input type.")

        local_num_nodes[node_type] = len(ndf)
        node_offset_tensor = torch.zeros(
            (world_size,), dtype=torch.int64, device=device
        )
        current_num_nodes = torch.tensor([len(ndf)], dtype=torch.int64, device=device)

        torch.distributed.all_gather_into_tensor(node_offset_tensor, current_num_nodes)

        map_tensor = [
            torch.zeros((2, node_offset_tensor[i]), device=device, dtype=torch.int64)
            for i in range(node_offset_tensor.numel())
        ]

        node_offset_tensor = node_offset_tensor.cumsum(0).cpu()
        global_num_nodes[node_type] = int(node_offset_tensor[-1])
        local_node_offsets[node_type] = (
            0 if global_rank == 0 else int(node_offset_tensor[global_rank - 1])
        )

        local_renumber_map = torch.stack(
            [
                torch.arange(
                    local_node_offsets[node_type],
                    local_node_offsets[node_type] + local_num_nodes[node_type],
                ),
                torch.as_tensor(
                    ndf[args.node_colname], device="cpu", dtype=torch.int64
                ),
            ]
        )

        torch.distributed.all_gather(map_tensor, local_renumber_map.to(device))
        map_tensor = torch.concat(map_tensor, dim=1).cpu()
        global_renumber_map[node_type] = pandas.DataFrame(
            {
                "id": map_tensor[0].numpy(),
            },
            index=map_tensor[1].numpy(),
        )

        local_renumber_map_df = pandas.DataFrame(
            {"id": local_renumber_map[0].numpy()}, index=local_renumber_map[1].numpy()
        )

        if args.output_format.lower() == "csv":
            local_renumber_map_df.to_csv(
                os.path.join(output_folder_name, f"{node_fname}_renumbered.csv"),
                index=False,
            )
        elif args.output_format.lower() == "parquet":
            local_renumber_map_df.to_parquet(
                os.path.join(output_folder_name, f"{node_fname}_renumbered.parquet"),
                index=False,
            )
        else:
            raise ValueError("Invalid output format.")

    edge_types = args.edge_types.split(";")

    for edge_type, edge_folder_name, output_folder_name in zip(
        edge_types,
        args.edge_input_folders.split(","),
        args.edge_output_folders.split(","),
    ):
        edge_type = tuple(edge_type.split(","))
        src_type, rel_type, dst_type = edge_type

        edge_fname = os.listdir(edge_folder_name)[local_rank]
        edge_fpath = os.path.join(edge_folder_name, edge_fname)

        edf = pandas.read_csv(edge_fpath)
        srcs = edf[args.source_colname].values
        dsts = edf[args.destination_colname].values

        src_map = global_renumber_map[src_type]["id"]
        dst_map = global_renumber_map[dst_type]["id"]

        new_edf = pandas.DataFrame(
            {
                args.source_colname: src_map.loc[srcs].values,
                args.destination_colname: dst_map.loc[dsts].values,
            }
        )

        if args.output_format.lower() == "parquet":
            new_edf.to_parquet(
                os.path.join(output_folder_name, f"{edge_fname}_renumbered.parquet"),
                index=False,
            )
        elif args.output_format.lower() == "csv":
            new_edf.to_csv(
                os.path.join(output_folder_name, f"{edge_fname}_renumbered.csv"),
                index=False,
            )
        else:
            raise ValueError("Invalid output format.")
