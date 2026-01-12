#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

# Only use stable ABI package for Python >= 3.11
if [[ "${RAPIDS_PY_VERSION}" != "3.10" ]]; then
  source ./ci/use_upstream_sabi_wheels.sh
fi

package_dir="python/pylibwholegraph"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download the libwholegraph wheel built in the previous step and make it
# available for pip to find.
#
# Using env variable PIP_CONSTRAINT (initialized by 'rapids-init-pip') is necessary to ensure the constraints
# are used when creating the isolated build environment.
LIBWHOLEGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libwholegraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
echo "libwholegraph-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBWHOLEGRAPH_WHEELHOUSE}"/libwholegraph_*.whl)" >> "${PIP_CONSTRAINT}"

export SKBUILD_CMAKE_ARGS="-DBUILD_SHARED_LIBS=ON;-DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE"

./ci/build_wheel.sh pylibwholegraph ${package_dir} python
./ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

# Only use stable ABI package naming for Python >= 3.11
if [[ "${RAPIDS_PY_VERSION}" != "3.10" ]]; then
  RAPIDS_PACKAGE_NAME="$(rapids-package-name wheel_python pylibwholegraph --stable --cuda)"
  export RAPIDS_PACKAGE_NAME
fi
