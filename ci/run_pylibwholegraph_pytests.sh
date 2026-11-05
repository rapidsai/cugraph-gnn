#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if grep -qi microsoft /proc/version; then export NCCL_CUMEM_ENABLE=0; fi

# Support invoking run_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/pylibwholegraph/pylibwholegraph/

df -h /dev/shm
for f in tests/pylibwholegraph/test_wholememory_binding.py tests/pylibwholegraph/test_wholememory_tensor.py tests/wholegraph_torch/ops/test_wholegraph_gather_scatter.py tests/wholegraph_torch/ops/test_wholegraph_unweighted_sample_without_replacement.py tests/pylibwholegraph/test_tensor_file_formats_io.py; do echo "=== $f"; NCCL_DEBUG=WARN timeout 900 pytest -x -s -v -rs --cache-clear --import-mode=append "$f" 2>&1 | tail -150 || true; done
exit 1
