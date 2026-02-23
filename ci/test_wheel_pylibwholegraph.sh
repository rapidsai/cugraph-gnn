#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -e -u -o pipefail

# Delete system libnccl.so to ensure the wheel is used.
# (but only do this in CI, to avoid breaking local dev environments)
if [[ "${CI:-}" == "true" ]]; then
  rm -rf /usr/lib64/libnccl*
fi

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
LIBWHOLEGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libwholegraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
PYLIBWHOLEGRAPH_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" pylibwholegraph --stable --cuda "$RAPIDS_CUDA_VERSION")")

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

# generate constraints, accounting for 'oldset' and 'latest' dependencies
rapids-dependency-file-generator \
    --output requirements \
    --file-key "test_pylibwholegraph" \
    --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES};include_torch_extra_index=false" \
| tee "${PIP_CONSTRAINT}"

# ensure a CUDA variant of 'torch' is used
./ci/install-torch-cuda-variant.sh

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-logger "Installing Packages"
rapids-pip-retry install \
    --constraint "${PIP_CONSTRAINT}" \
    "$(echo "${PYLIBWHOLEGRAPH_WHEELHOUSE}"/pylibwholegraph*.whl)[test]" \
    "${LIBWHOLEGRAPH_WHEELHOUSE}"/*.whl

rapids-logger "pytest pylibwholegraph"
cd python/pylibwholegraph/pylibwholegraph/tests
python -m pytest \
  --cache-clear \
  --forked \
  --import-mode=append \
  .
