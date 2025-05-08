#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

if [ "$RAPIDS_PY_VERSION" != "3.13" ]; then
  package_dir="python/cugraph-dgl"

  ./ci/build_wheel.sh cugraph-dgl ${package_dir} python
  ./ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
else
  rapids-logger "Not building DGL wheel on Python 3.13"
fi
