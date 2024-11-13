#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/cugraph-pyg"

./ci/build_wheel.sh cugraph-pyg ${package_dir}
./ci/validate_wheel.sh ${package_dir} dist
