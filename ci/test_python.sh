#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking test_python.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

if [[ "${RAPIDS_CUDA_VERSION%%.*}" == "11" ]]; then
  DGL_CHANNEL="dglteam/label/th24_cu118"
else
  DGL_CHANNEL="dglteam/label/th24_cu124"
fi

. /opt/conda/etc/profile.d/conda.sh

RAPIDS_VERSION="$(rapids-version)"

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

# RAPIDS_DATASET_ROOT_DIR is used by test scripts
export RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"
mkdir -p "${RAPIDS_DATASET_ROOT_DIR}"
pushd "${RAPIDS_DATASET_ROOT_DIR}"
./get_test_data.sh --benchmark
popd

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Test runs that include tests that use dask require
# --import-mode=append. Those tests start a LocalCUDACluster that inherits
# changes from pytest's modifications to PYTHONPATH (which defaults to
# prepending source tree paths to PYTHONPATH).  This causes the
# LocalCUDACluster subprocess to import cugraph from the source tree instead of
# the install location, and in most cases, the source tree does not have
# extensions built in-place and will result in ImportErrors.
#
# FIXME: TEMPORARILY disable MG PropertyGraph tests (experimental) tests and
# bulk sampler IO tests (hangs in CI)

if [[ "${RUNNER_ARCH}" != "ARM64" ]]; then
  rapids-logger "(cugraph-dgl) Generate Python testing dependencies"
  rapids-dependency-file-generator \
    --output conda \
    --file-key test_cugraph_dgl \
    --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}"  \
    --prepend-channel "${CPP_CHANNEL}" \
    --prepend-channel "${PYTHON_CHANNEL}" \
    --prepend-channel conda-forge \
    --prepend-channel "${DGL_CHANNEL}" \
    --prepend-channel nvidia \
  | tee env.yaml

  rapids-mamba-retry env create --yes -f env.yaml -n test_cugraph_dgl

  # activate test_cugraph_dgl environment for dgl
  set +u
  conda activate test_cugraph_dgl
  set -u


  rapids-print-env

  rapids-logger "Check GPU usage"
  nvidia-smi

  rapids-logger "pytest cugraph_dgl (single GPU)"
  ./ci/run_cugraph_dgl_pytests.sh \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cugraph-dgl.xml" \
    --cov-config=../../.coveragerc \
    --cov=cugraph_dgl \
    --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cugraph-dgl-coverage.xml" \
    --cov-report=term

  # Reactivate the test environment back
  set +u
  conda deactivate
  set -u
else
  rapids-logger "skipping cugraph_dgl pytest on ARM64"
fi

if [[ "${RUNNER_ARCH}" != "ARM64" ]]; then
  rapids-logger "(cugraph-pyg) Generate Python testing dependencies"
  rapids-dependency-file-generator \
    --output conda \
    --file-key test_cugraph_pyg \
    --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}"  \
    --prepend-channel "${CPP_CHANNEL}" \
    --prepend-channel "${PYTHON_CHANNEL}"
  | tee env.yaml

  rapids-mamba-retry env create --yes -f env.yaml -n test_cugraph_pyg

  # Temporarily allow unbound variables for conda activation.
  set +u
  conda activate test_cugraph_pyg
  set -u

  rapids-print-env

  rapids-logger "Check GPU usage"
  nvidia-smi

  rapids-logger "pytest cugraph_pyg (single GPU)"
  ./ci/run_cugraph_pyg_pytests.sh \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cugraph-pyg.xml" \
    --cov-config=../../.coveragerc \
    --cov=cugraph_pyg \
    --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cugraph-pyg-coverage.xml" \
    --cov-report=term

  # Reactivate the test environment back
  set +u
  conda deactivate
  set -u
else
  rapids-logger "skipping cugraph_pyg pytest on ARM64"
fi

if [[ "${RUNNER_ARCH}" != "ARM64" ]]; then
  rapids-logger "(pylibwholegraph) Generate Python testing dependencies"
  rapids-dependency-file-generator \
    --output conda \
    --file-key test_pylibwholegraph \
    --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}"  \
    --prepend-channel "${CPP_CHANNEL}" \
    --prepend-channel "${PYTHON_CHANNEL}" \
  | tee env.yaml

  rapids-mamba-retry env create --yes -f env.yaml -n test_pylibwholegraph

  # Temporarily allow unbound variables for conda activation.
  set +u
  conda activate test_pylibwholegraph
  set -u

  rapids-print-env

  rapids-logger "Check GPU usage"
  nvidia-smi

  rapids-logger "pytest pylibwholegraph (single GPU)"
  ./ci/run_pylibwholegraph_pytests.sh \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-pylibwholegraph.xml" \
    --cov-config=../../.coveragerc \
    --cov=pylibwholegraph \
    --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/pylibwholegraph-coverage.xml" \
    --cov-report=term

  # Reactivate the test environment back
  set +u
  conda deactivate
  set -u
else
  rapids-logger "skipping pylibwholegraph pytest on ARM64"
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
