#!/bin/bash
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Any failing command will set EXITCODE to non-zero
set -e           # abort the script on error, this will change for running tests (see below)
set -o pipefail  # piped commands propagate their error
set -E           # ERR traps are inherited by subcommands
trap "EXITCODE=1" ERR

NUMARGS=$#
ARGS=$*
THISDIR=$(cd $(dirname $0);pwd)
CUGRAPH_ROOT=$(cd ${THISDIR}/..;pwd)
GTEST_ARGS="--gtest_output=xml:${CUGRAPH_ROOT}/test-results/"
DOWNLOAD_MODE=""
EXITCODE=0

export RAPIDS_DATASET_ROOT_DIR=${RAPIDS_DATASET_ROOT_DIR:-${CUGRAPH_ROOT}/datasets}

# FIXME: consider using getopts for option parsing
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Add options unique to running a "quick" subset of tests here:
#  - pass --subset flag to download script to skip large downloads
#  - filter the "huge" dataset tests
if hasArg "--quick"; then
    echo "Running \"quick\" tests only..."
    DOWNLOAD_MODE="--subset"
    GTEST_FILTER="--gtest_filter=-hibench_test/Tests_MGSpmv_hibench.CheckFP32_hibench*:*huge*"
else
    echo "Running all tests..."
    # FIXME: do we still need to always filter these tests?
    GTEST_FILTER="--gtest_filter=-hibench_test/Tests_MGSpmv_hibench.CheckFP32_hibench*"
fi

if hasArg "--skip-download"; then
    echo "Using datasets in ${RAPIDS_DATASET_ROOT_DIR}"
else
    echo "Download datasets..."
    cd ${RAPIDS_DATASET_ROOT_DIR}
    bash ./get_test_data.sh ${DOWNLOAD_MODE}
fi

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    cd ${CUGRAPH_ROOT}/cpp/build
fi

# Do not abort the script on error from this point on. This allows all tests to
# run regardless of pass/fail, but relies on the ERR trap above to manage the
# EXITCODE for the script.
set +e

if hasArg "--run-cpp-tests"; then
    # wholegraph tests, presumably
fi

if hasArg "--run-python-tests"; then

    echo "Python pytest for cugraph_pyg (single-GPU only)..."
    conda list
    cd ${CUGRAPH_ROOT}/python/cugraph-pyg/cugraph_pyg
    # rmat is not tested because of MG testing
    pytest -sv -m sg --cache-clear --junitxml=${CUGRAPH_ROOT}/junit-cugraph-pytests.xml -v --cov-config=.coveragerc --cov=cugraph_pyg --cov-report=xml:${WORKSPACE}/python/cugraph_pyg/cugraph-coverage.xml --cov-report term --ignore=raft --benchmark-disable
    echo "Ran Python pytest for cugraph_pyg : return code was: $?, test script exit code is now: $EXITCODE"

fi

echo "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
