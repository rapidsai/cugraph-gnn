#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/pylibwholegraph"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download the libcugraph wheel built in the previous step and make it
# available for pip to find.
RAPIDS_PY_WHEEL_NAME="libwholegraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp /tmp/libwholegraph_dist
echo "libwholegraph-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo /tmp/libwholegraph_dist/libwholegraph_*.whl)" > /tmp/constraints.txt

# Using env variable PIP_CONSTRAINT is necessary to ensure the constraints
# are used when creating the isolated build environment.
export PIP_CONSTRAINT="/tmp/constraints.txt"

export SKBUILD_CMAKE_ARGS="-DBUILD_SHARED_LIBS=ON;-DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE;-DCUDA_STATIC_RUNTIME=ON;-DWHOLEGRAPH_BUILD_WHEELS=ON"

./ci/build_wheel.sh pylibwholegraph ${package_dir} python
./ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
