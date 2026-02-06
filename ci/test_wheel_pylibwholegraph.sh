#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -e          # abort the script on error
set -o pipefail # piped commands propagate their error
set -E          # ERR traps are inherited by subcommands

# Delete system libnccl.so to ensure the wheel is used.
# (but only do this in CI, to avoid breaking local dev environments)
if [[ "${CI:-}" == "true" ]]; then
  rm -rf /usr/lib64/libnccl*
fi

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
LIBWHOLEGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libwholegraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
PYLIBWHOLEGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="pylibwholegraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)

# determine pytorch source
if [[ "${CUDA_MAJOR}" == "12" ]]; then
  PYTORCH_INDEX="https://download.pytorch.org/whl/cu126"
else
  PYTORCH_INDEX="https://download.pytorch.org/whl/cu130"
fi
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-logger "Installing Packages"
rapids-pip-retry install \
    --extra-index-url ${PYTORCH_INDEX} \
    "$(echo "${PYLIBWHOLEGRAPH_WHEELHOUSE}"/pylibwholegraph*.whl)[test]" \
    "${LIBWHOLEGRAPH_WHEELHOUSE}"/*.whl \
    'torch>=2.3'

rapids-logger "pytest pylibwholegraph"
cd python/pylibwholegraph/pylibwholegraph/tests
python -m pytest \
  --cache-clear \
  --forked \
  --import-mode=append \
  .
