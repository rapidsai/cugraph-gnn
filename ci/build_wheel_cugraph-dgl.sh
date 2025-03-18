#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/cugraph-dgl"

wheel_dir=${RAPIDS_WHEEL_BLD_OUTPUT_DIR:-"final_dist"}

./ci/build_wheel.sh cugraph-dgl ${package_dir}
./ci/validate_wheel.sh ${package_dir} "${wheel_dir}"
