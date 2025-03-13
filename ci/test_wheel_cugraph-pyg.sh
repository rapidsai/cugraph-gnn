#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -eoxu pipefail

package_name="cugraph-pyg"

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# Download the pylibwholegraph and cugraph-pyg built in the previous step
RAPIDS_PY_WHEEL_NAME="pylibwholegraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./local-deps
RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-s3 python ./dist

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install \
    -v \
    "$(echo ./local-deps/pylibwholegraph_${RAPIDS_PY_CUDA_SUFFIX}*.whl)" \
    "$(echo ./dist/cugraph_pyg_${RAPIDS_PY_CUDA_SUFFIX}*.whl)[test]"

# RAPIDS_DATASET_ROOT_DIR is used by test scripts
export RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"

rapids-logger "pytest cugraph-pyg (single GPU)"
pushd python/cugraph-pyg/cugraph_pyg
python -m pytest \
  --cache-clear \
  --benchmark-disable \
  tests
# Test examples (disabled due to excessive network bandwidth usage)
#for e in "$(pwd)"/examples/*.py; do
#  rapids-logger "running example $e"
#  (yes || true) | python $e
#done
popd
