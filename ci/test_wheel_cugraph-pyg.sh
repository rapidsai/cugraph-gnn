#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -eoxu pipefail

source rapids-init-pip

package_name="cugraph-pyg"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# Download the libwholegraph, pylibwholegraph, and cugraph-pyg built in the previous step
LIBWHOLEGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libwholegraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
PYLIBWHOLEGRAPH_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="pylibwholegraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
CUGRAPH_PYG_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-github python)

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install \
    -v \
    "${LIBWHOLEGRAPH_WHEELHOUSE}"/*.whl \
    "$(echo "${PYLIBWHOLEGRAPH_WHEELHOUSE}"/pylibwholegraph_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)" \
    "$(echo "${CUGRAPH_PYG_WHEELHOUSE}"/cugraph_pyg_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

# RAPIDS_DATASET_ROOT_DIR is used by test scripts
export RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"
mkdir -p "${RAPIDS_DATASET_ROOT_DIR}"
pushd "${RAPIDS_DATASET_ROOT_DIR}"
./get_test_data.sh --test
popd

# Enable legacy behavior of torch.load for examples relying on ogb
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

rapids-logger "pytest cugraph-pyg (single GPU)"
pushd python/cugraph-pyg/cugraph_pyg
python -m pytest \
  --cache-clear \
  --benchmark-disable \
  tests

# Test examples (disabled due to lack of memory)
#for e in "$(pwd)"/examples/*.py; do
#  rapids-logger "running example $e"
#  (yes || true) | python -m torch.distributed.run --nnodes 1 --nproc_per_node 1 $e --dataset_root "${RAPIDS_DATASET_ROOT_DIR}/ogb_datasets"
#done

# rapids-logger "running bitcoin example"
# (yes || true) | python -m torch.distributed.run --nnodes 1 --nproc_per_node 1 "$(pwd)"/examples/fraud/bitcoin_mnmg.py --dataset_root "${RAPIDS_DATASET_ROOT_DIR}" --embedding_dir "${RAPIDS_DATASET_ROOT_DIR}/bitcoin_embeddings"
# python "$(pwd)"/examples/fraud/bitcoin_rf.py --dataset_root "${RAPIDS_DATASET_ROOT_DIR}" --embedding_dir "${RAPIDS_DATASET_ROOT_DIR}/bitcoin_embeddings"

popd
