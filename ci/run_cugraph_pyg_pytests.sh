#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking run_cugraph_pyg_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cugraph-pyg/cugraph_pyg

export PYTHONUNBUFFERED=1
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gdb) > /tmp/apt.log 2>&1 || tail -5 /tmp/apt.log
pip install -q py-spy > /tmp/pyspy.log 2>&1 || tail -5 /tmp/pyspy.log
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,ENV,SHM,PROXY NCCL_DEBUG_FILE=/tmp/nccl.%p.log pytest -v -rs --cache-clear --benchmark-disable . > /tmp/pytest.out 2>&1 &
PID=$!
sleep 240
echo "=== pytest.out"; grep -vE "PASSED|SKIPPED" /tmp/pytest.out | head -40
echo "=== processes"; ps -ef --forest | grep -v "ps -ef" | tail -20
echo "=== gdb"; timeout 120 gdb -p "$PID" -batch -ex "set pagination off" -ex "thread apply all bt 30" 2>&1 | grep -vE "^\[New LWP|^$" | head -300 || true
echo "=== py-spy"; timeout 120 py-spy dump --native --pid "$PID" 2>&1 | head -200 || true
echo "=== nccl logs"; for f in /tmp/nccl.*.log; do echo "--- $f"; grep -vE "Trees|Channel|P2P" "$f" | tail -40; done
kill -9 "$PID" || true
exit 1

# Used to skip certain examples in CI due to memory limitations
export CI=true

# Enable legacy behavior of torch.load for examples relying on ogb
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# Test examples (disabled due to lack of memory)
#for e in "$(pwd)"/examples/*.py; do
#  echo "running example $e"
#  (yes || true) | torchrun --nnodes 1 --nproc_per_node 1 $e --dataset_root "${RAPIDS_DATASET_ROOT_DIR}/ogb_datasets"
#done

# echo "running bitcoin example"
# (yes || true) | torchrun --nnodes 1 --nproc_per_node 1 "$(pwd)"/examples/fraud/bitcoin_mnmg.py --dataset_root "${RAPIDS_DATASET_ROOT_DIR}" --embedding_dir "${RAPIDS_DATASET_ROOT_DIR}/bitcoin_embeddings"
# python "$(pwd)"/examples/fraud/bitcoin_rf.py --dataset_root "${RAPIDS_DATASET_ROOT_DIR}" --embedding_dir "${RAPIDS_DATASET_ROOT_DIR}/bitcoin_embeddings"
