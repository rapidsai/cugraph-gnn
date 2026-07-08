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

LIBWHOLEGRAPH_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libwholegraph cugraph-gnn --cuda "$RAPIDS_CUDA_VERSION")")
PYLIBWHOLEGRAPH_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python pylibwholegraph cugraph-gnn --stable --cuda "$RAPIDS_CUDA_VERSION")")

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
PIP_INSTALL_ARGS+=("${TORCH_WHEEL_DIR}"/torch-*.whl)

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-logger "Installing Packages"
rapids-pip-retry install \
    "${PIP_INSTALL_ARGS[@]}"

# 'torch' is an optional dependency of 'pylibwholegraph'... confirm that it's actually
# installed here and that we've installed a package with CUDA support.
rapids-logger "Confirming that PyTorch is installed"
python -c "import torch; assert torch.cuda.is_available()"

rapids-logger "pytest pylibwholegraph (with 'torch')"
./ci/run_pylibwholegraph_pytests.sh \
  --cov-config=../../.coveragerc \
  --cov=pylibwholegraph \
  --cov-fail-under=15

rapids-logger "import pylibwholegraph (no 'torch')"
./ci/uninstall-torch-wheels.sh

python -c "import pylibwholegraph; print(f'pylibwholegraph version: {pylibwholegraph.__version__}')"

rapids-logger "pytest pylibwholegraph (no 'torch')"
./ci/run_pylibwholegraph_pytests.sh
