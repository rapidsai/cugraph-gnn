#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

package_dir="python/pylibwholegraph"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download the libwholegraph wheel built in the previous step and make it
# available for pip to find.
#
# env variable 'PIP_CONSTRAINT' is set up by rapids-init-pip. It constrains all subsequent
# 'pip install', 'pip download', etc. calls (except those used in 'pip wheel', handled separately in build scripts)
LIBWHOLEGRAPH_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libwholegraph cugraph-gnn --cuda "$RAPIDS_CUDA_VERSION")")
echo "libwholegraph-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBWHOLEGRAPH_WHEELHOUSE}"/libwholegraph_*.whl)" >> "${PIP_CONSTRAINT}"

export SKBUILD_CMAKE_ARGS="-DBUILD_SHARED_LIBS=ON;-DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE"

# TODO: move this variable into `ci-wheel`
# Format Python limited API version string
RAPIDS_PY_API="cp${RAPIDS_PY_VERSION//./}"
export RAPIDS_PY_API

./ci/build_wheel.sh pylibwholegraph ${package_dir} --stable
./ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PACKAGE_NAME="$(rapids-artifact-name wheel_python pylibwholegraph cugraph-gnn --stable --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
