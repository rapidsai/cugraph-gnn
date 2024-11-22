#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -eoxu pipefail

package_name="cugraph-dgl"

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# Download the pylibwholegraph and cugraph-dgl built in the previous step
RAPIDS_PY_WHEEL_NAME="pylibwholegraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-deps
RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-s3 ./dist

# determine pytorch and DGL sources
PKG_CUDA_VER="$(echo ${CUDA_VERSION} | cut -d '.' -f1,2 | tr -d '.')"
PKG_CUDA_VER_MAJOR=${PKG_CUDA_VER:0:2}
if [[ "${PKG_CUDA_VER_MAJOR}" == "12" ]]; then
  PYTORCH_CUDA_VER="121"
else
  PYTORCH_CUDA_VER=$PKG_CUDA_VER
fi
PYTORCH_URL="https://download.pytorch.org/whl/cu${PYTORCH_CUDA_VER}"
DGL_URL="https://data.dgl.ai/wheels/torch-2.3/cu${PYTORCH_CUDA_VER}/repo.html"

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install \
    -v \
    --extra-index-url "${PYTORCH_URL}" \
    --find-links "${DGL_URL}" \
    "$(echo ./local-deps/pylibwholegraph_${RAPIDS_PY_CUDA_SUFFIX}*.whl)" \
    "$(echo ./dist/cugraph_dgl_${RAPIDS_PY_CUDA_SUFFIX}*.whl)[test]" \
    'dgl==2.4.0' \
    'torch>=2.3'

# RAPIDS_DATASET_ROOT_DIR is used by test scripts
export RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"

python -m pytest python/cugraph-dgl/cugraph_dgl/tests
