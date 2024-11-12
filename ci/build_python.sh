#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

# TODO: revert this once we start publishing nightly packages
#       from the 'cugraph-gnn' repo and stop publishing them from
#       the 'cugraph' / 'wholegraph' repos
# rapids-generate-version > ./VERSION
echo "24.12.00a1000" > ./VERSION

sccache --zero-stats

# TODO: Remove `--no-test` flags once importing on a CPU
# node works correctly
rapids-logger "Begin pylibwholegraph build"
RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/pylibwholegraph

sccache --show-adv-stats

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/cugraph-pyg

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/cugraph-dgl

rapids-upload-conda-to-s3 python
