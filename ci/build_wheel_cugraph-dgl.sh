#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/cugraph-dgl"

# Download the libcuml wheel built in the previous step and make it
# available for pip to find.
LIBCUML_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp /tmp/libcuml_dist)

source ./ci/use_wheels_from_prs.sh

./ci/build_wheel.sh cugraph-dgl ${package_dir}
./ci/validate_wheel.sh ${package_dir} dist
