#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking test_python.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

. /opt/conda/etc/profile.d/conda.sh

RAPIDS_VERSION="$(rapids-version)"

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

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
  rapids-mamba-retry env create --yes -f env.yaml -n test_cugraph_dgl

  # activate test_cugraph_dgl environment for dgl
  set +u
  conda activate test_cugraph_dgl
  set -u

  rapids-mamba-retry install \
    --channel "${CPP_CHANNEL}" \
    --channel "${PYTHON_CHANNEL}" \
    --channel pytorch \
    --channel conda-forge \
    --channel dglteam/label/th23_cu118 \
    --channel nvidia \
    "pylibwholegraph=${RAPIDS_VERSION}.*" \
    "pylibcugraphops=${RAPIDS_VERSION}.*" \
    "cugraph=${RAPIDS_VERSION}.*" \
    "cugraph-dgl=${RAPIDS_VERSION}.*" \
    'pytorch::pytorch>=2.3,<2.4' \
    'cuda-version=11.8' \
    "ogb"

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
  conda activate test
  set -u
else
  rapids-logger "skipping cugraph_dgl pytest on ARM64"
fi

if [[ "${RUNNER_ARCH}" != "ARM64" ]]; then
  rapids-mamba-retry env create --yes -f env.yaml -n test_cugraph_pyg

  # Temporarily allow unbound variables for conda activation.
  set +u
  conda activate test_cugraph_pyg
  set -u

  # TODO re-enable logic once CUDA 12 is testable
  #if [[ "${RAPIDS_CUDA_VERSION}" == "11.8.0" ]]; then
  CONDA_CUDA_VERSION="11.8"
  PYG_URL="https://data.pyg.org/whl/torch-2.1.0+cu118.html"
  #else
  #  CONDA_CUDA_VERSION="12.1"
  #  PYG_URL="https://data.pyg.org/whl/torch-2.1.0+cu121.html"
  #fi

  # Will automatically install built dependencies of cuGraph-PyG
  rapids-mamba-retry install \
    --channel "${CPP_CHANNEL}" \
    --channel "${PYTHON_CHANNEL}" \
    --channel pytorch \
    "pylibwholegraph=${RAPIDS_VERSION}.*" \
    "pylibcugraphops=${RAPIDS_VERSION}.*" \
    "cugraph=${RAPIDS_VERSION}.*" \
    "cugraph-pyg=${RAPIDS_VERSION}.*" \
    "pytorch::pytorch>=2.3,<2.4" \
    "ogb"

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
  conda activate test
  set -u
else
  rapids-logger "skipping cugraph_pyg pytest on ARM64"
fi

if [[ "${RUNNER_ARCH}" != "ARM64" ]]; then
  rapids-mamba-retry env create --yes -f env.yaml -n test_pylibwholegraph

  # Temporarily allow unbound variables for conda activation.
  set +u
  conda activate test_pylibwholegraph
  set -u

  # TODO re-enable logic once CUDA 12 is testable
  #if [[ "${RAPIDS_CUDA_VERSION}" == "11.8.0" ]]; then
  CONDA_CUDA_VERSION="11.8"
  #else
  #  CONDA_CUDA_VERSION="12.1"
  #fi

  # Will automatically install built dependencies of pylibwholegraph
  rapids-mamba-retry install \
    --channel "${CPP_CHANNEL}" \
    --channel "${PYTHON_CHANNEL}" \
    --channel pytorch \
    "pylibwholegraph=${RAPIDS_VERSION}.*" \
    "pytorch::pytorch>=2.3,<2.4" \
    "ogb"

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
  conda activate test
  set -u
else
  rapids-logger "skipping cugraph_pyg pytest on ARM64"
fi


rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
