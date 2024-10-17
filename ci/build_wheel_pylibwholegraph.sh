
#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

export SKBUILD_CMAKE_ARGS="-DDETECT_CONDA_ENV=OFF;-DBUILD_SHARED_LIBS=OFF;-DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE;-DCUDA_STATIC_RUNTIME=ON;-DWHOLEGRAPH_BUILD_WHEELS=ON"

./ci/build_wheel.sh pylibwholegraph python/pylibwholegraph
