#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

source rapids-init-pip

package_dir="python/libwholegraph"

export SKBUILD_CMAKE_ARGS="-DBUILD_SHARED_LIBS=ON;-DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE;-DCUDA_STATIC_RUNTIME=ON"

./ci/build_wheel.sh libwholegraph ${package_dir} cpp
./ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
