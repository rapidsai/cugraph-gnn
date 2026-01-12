#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

# Using env variable PIP_CONSTRAINT (initialized by 'rapids-init-pip') is necessary to ensure the constraints
# are used when creating the isolated build environment.
if [[ "${RAPIDS_PY_VERSION}" != "3.10" ]]; then
  PYLIBWHOLEGRAPH_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" pylibwholegraph --stable --cuda "$RAPIDS_CUDA_VERSION")")
  source ./ci/use_upstream_sabi_wheels.sh
else
  PYLIBWHOLEGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="pylibwholegraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
fi

LIBWHOLEGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libwholegraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)

package_dir="python/cugraph-pyg"

cat >> "${PIP_CONSTRAINT}" <<EOF
libwholegraph-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBWHOLEGRAPH_WHEELHOUSE}"/libwholegraph_*.whl)
pylibwholegraph-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${PYLIBWHOLEGRAPH_WHEELHOUSE}"/pylibwholegraph_*.whl)
EOF

./ci/build_wheel.sh cugraph-pyg ${package_dir} python
./ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
