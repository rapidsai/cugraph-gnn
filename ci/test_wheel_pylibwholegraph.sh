#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

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

# generate constraints (possibly pinning to oldest support versions of dependencies)
rapids-generate-pip-constraints test_pylibwholegraph "${PIP_CONSTRAINT}"

PIP_INSTALL_ARGS=(
  --prefer-binary
  --constraint "${PIP_CONSTRAINT}"
  "$(echo "${PYLIBWHOLEGRAPH_WHEELHOUSE}"/pylibwholegraph*.whl)[test]"
  "${LIBWHOLEGRAPH_WHEELHOUSE}"/*.whl
)

# ensure a CUDA variant of 'torch' is used (if one is available)
TORCH_WHEEL_DIR="$(mktemp -d)"
./ci/download-torch-wheels.sh "${TORCH_WHEEL_DIR}"

# 'cugraph-pyg' is still expected to be importable
# and testable in an environment where 'torch' isn't installed.
torch_installed=true
if [ -z "$(ls -A ${TORCH_WHEEL_DIR} 2>/dev/null)" ]; then
  rapids-echo-stderr "No 'torch' wheels downloaded."
  torch_installed=false
else
  PIP_INSTALL_ARGS+=("${TORCH_WHEEL_DIR}"/torch-*.whl)
fi

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-logger "Installing Packages"
rapids-pip-retry install \
    "${PIP_INSTALL_ARGS[@]}"

if [[ "${torch_installed}" == "true" ]]; then
  rapids-logger "pytest pylibwholegraph (with 'torch')"
  ./ci/run_pylibwholegraph_pytests.sh
fi

rapids-logger "pytest pylibwholegraph (no 'torch')"
pip uninstall --yes 'torch'
python -c "import pylibwholegraph; print(pylibwholegraph.__version__)"
./ci/run_pylibwholegraph_pytests.sh
