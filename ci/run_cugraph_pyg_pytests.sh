#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking run_cugraph_pyg_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cugraph-pyg/cugraph_pyg

pytest --cache-clear --benchmark-disable "$@" .

# Used to skip certain examples in CI due to memory limitations
export CI=true

# Test examples (disabled due to excessive network bandwidth usage)
for e in "$(pwd)"/examples/*.py; do
  rapids-logger "running example $e"
  (yes || true) | torchrun --nnodes 1 --nproc_per_node 1 $e --dataset_root "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../datasets
done
