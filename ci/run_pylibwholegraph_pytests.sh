#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking run_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/pylibwholegraph/pylibwholegraph/

python /__w/cugraph-gnn/cugraph-gnn/ci/cumem_probe.py || true
df -h /dev/shm
for v in 0; do echo "=== NCCL_CUMEM_ENABLE=$v NCCL_MAX_NCHANNELS=4"; NCCL_MAX_NCHANNELS=4 NCCL_DEBUG=WARN NCCL_CUMEM_ENABLE=$v timeout 300 pytest -rs --cache-clear --import-mode=append -x "$@" tests/pylibwholegraph/test_wholememory_binding.py 2>&1 | tail -30 || true; done
exit 1
pytest -rs --cache-clear --import-mode=append "$@" tests
