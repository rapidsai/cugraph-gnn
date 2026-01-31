#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

RAPIDS_INIT_PIP_REMOVE_NVIDIA_INDEX="true"
export RAPIDS_INIT_PIP_REMOVE_NVIDIA_INDEX
source rapids-init-pip

package_dir="python/pylibwholegraph"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download the libwholegraph wheel built in the previous step and make it
# available for pip to find.
#
# env variable 'PIP_CONSTRAINT' is set up by rapids-init-pip. It constrains all subsequent
# 'pip install', 'pip download', etc. calls (except those used in 'pip wheel', handled separately in build scripts)
LIBWHOLEGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libwholegraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
echo "libwholegraph-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBWHOLEGRAPH_WHEELHOUSE}"/libwholegraph_*.whl)" >> "${PIP_CONSTRAINT}"

export SKBUILD_CMAKE_ARGS="-DBUILD_SHARED_LIBS=ON;-DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE"

./ci/build_wheel.sh pylibwholegraph ${package_dir} python
./ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
