#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -e          # abort the script on error
set -o pipefail # piped commands propagate their error
set -E          # ERR traps are inherited by subcommands

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="libwholegraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./local-libwholegraph-dep
RAPIDS_PY_WHEEL_NAME="pylibwholegraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist

# determine pytorch source
PKG_CUDA_VER="$(echo ${CUDA_VERSION} | cut -d '.' -f1,2 | tr -d '.')"
PKG_CUDA_VER_MAJOR=${PKG_CUDA_VER:0:2}
if [[ "${PKG_CUDA_VER_MAJOR}" == "12" ]]; then
  INDEX_URL="https://download.pytorch.org/whl/cu121"
else
  INDEX_URL="https://download.pytorch.org/whl/cu${PKG_CUDA_VER}"
fi
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-logger "Installing Packages"
rapids-pip-retry install \
    --extra-index-url ${INDEX_URL} \
    "$(echo ./dist/pylibwholegraph*.whl)[test]" \
    ./local-libwholegraph-dep/*.whl \
    'torch>=2.3'

rapids-logger "pytest pylibwholegraph"
cd python/pylibwholegraph/pylibwholegraph/tests
python -m pytest \
  --cache-clear \
  --forked \
  --import-mode=append \
  .
