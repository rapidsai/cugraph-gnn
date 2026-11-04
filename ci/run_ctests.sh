#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support customizing the ctests' install location
# First, try the installed location (CI/conda environments)
installed_test_location="${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libwholegraph/"
# Fall back to the build directory (devcontainer environments)
devcontainers_test_location="$(dirname "$(realpath "${BASH_SOURCE[0]}")")/../cpp/build/latest"

if [[ -d "${installed_test_location}" ]]; then
    cd "${installed_test_location}"
elif [[ -d "${devcontainers_test_location}" ]]; then
    cd "${devcontainers_test_location}"
else
    echo "Error: Test location not found. Searched:" >&2
    echo "  - ${installed_test_location}" >&2
    echo "  - ${devcontainers_test_location}" >&2
    exit 1
fi

df -h /dev/shm; uname -r
for envs in "NCCL_CUMEM_ENABLE=0 NCCL_MAX_NCHANNELS=16" "NCCL_CUMEM_ENABLE=0 NCCL_MAX_NCHANNELS=4" "NCCL_CUMEM_ENABLE=1 NCCL_MAX_NCHANNELS=4"; do echo "=== $envs"; env $envs NCCL_DEBUG=WARN timeout 300 ./WHOLEGRAPH_EMBEDDING_TEST --gtest_filter='*EmbeddingGatherTest/3:*EmbeddingGatherTest/4' 2>&1 | grep -E "NCCL WARN|WholeMemory failure|OK \]|FAILED  \]|PASSED" || true; done
exit 1
find . -type f -executable -print0 | xargs -0 -r -t -n1 -P1 sh -c 'exec "$0"';
