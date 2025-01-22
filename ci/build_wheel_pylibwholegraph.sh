#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/pylibwholegraph"

export SKBUILD_CMAKE_ARGS="-DBUILD_SHARED_LIBS=OFF;-DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE;-DCUDA_STATIC_RUNTIME=ON;-DWHOLEGRAPH_BUILD_WHEELS=ON"

./ci/build_wheel.sh pylibwholegraph ${package_dir}
./ci/validate_wheel.sh ${package_dir} final_dist
