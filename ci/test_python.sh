#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking test_python.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

. /opt/conda/etc/profile.d/conda.sh

RAPIDS_VERSION="$(rapids-version)"

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}"  \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${PYTHON_CHANNEL}" \
| tee env.yaml

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

if [[ "${RUNNER_ARCH}" != "ARM64" ]]; then
  rapids-mamba-retry env create --yes -f env.yaml -n test_pylibwholegraph

  # Temporarily allow unbound variables for conda activation.
  set +u
  conda activate test_pylibwholegraph
  set -u

  # Will automatically install built dependencies of pylibwholegraph
  rapids-mamba-retry install \
    --channel "${CPP_CHANNEL}" \
    --channel "${PYTHON_CHANNEL}" \
    --channel pytorch \
    'mkl<2024.1.0' \
    "pylibwholegraph=${RAPIDS_VERSION}" \
    'pytorch::pytorch>=2.3,<2.4' \
    'pytest-forked' \
    'ogb'

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
