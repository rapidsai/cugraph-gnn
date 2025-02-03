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

torchrun \
    --nnodes 1 \
    --nproc-per-node 2 \
    renumber_kg.py \
        --node_types "paper,author" \
        --node_input_folders "/home/nfs/abarghi/test_renumber_kg/paper,/home/nfs/abarghi/test_renumber_kg/author" \
        --node_output_folders "/home/nfs/abarghi/test_renumber_kg/paper_renumbered,/home/nfs/abarghi/test_renumber_kg/author_renumbered" \
        --node_colname "ID" \
        --edge_types "paper,cites,paper;author,writes,paper" \
        --edge_input_folders "/home/nfs/abarghi/test_renumber_kg/paper_cites_paper,/home/nfs/abarghi/test_renumber_kg/author_writes_paper" \
        --edge_output_folders "/home/nfs/abarghi/test_renumber_kg/paper_cites_paper_renumbered,/home/nfs/abarghi/test_renumber_kg/author_writes_paper_renumbered" \
        --source_colname "SRC" \
        --destination_colname "DST" \
        --input_format "csv" \
        --output_format "csv" \
        --use_managed_memory
