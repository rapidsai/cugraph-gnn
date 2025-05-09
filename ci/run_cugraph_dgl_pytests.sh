#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail


if [ "$RAPIDS_PY_VERSION" != "3.13" ]; then
  # Support invoking run_cugraph_dgl_pytests.sh outside the script directory
  cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cugraph-dgl/cugraph_dgl

  pytest --cache-clear --ignore=mg "$@" .
else
  rapids-logger "Skipping DGL tests because Python version 3.13"
fi
