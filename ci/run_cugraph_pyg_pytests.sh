#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking run_cugraph_pyg_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cugraph-pyg/cugraph_pyg

export PYTHONUNBUFFERED=1
(apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq gdb procps) > /tmp/apt.log 2>&1 || tail -5 /tmp/apt.log
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT NCCL_DEBUG_FILE=/tmp/nccl.%p.log pytest -v -rs --cache-clear --benchmark-disable "$@" . > /tmp/pytest.out 2>&1 &
PID=$!
last=0; idle=0
while kill -0 "$PID" 2>/dev/null; do
  sleep 30
  sz=$(stat -c %s /tmp/pytest.out)
  if [ "$sz" = "$last" ]; then idle=$((idle+30)); else idle=0; last=$sz; fi
  if [ "$idle" -ge 420 ]; then
    echo "=== HANG"; tail -3 /tmp/pytest.out
    for p in "$PID" $(pgrep -P "$PID"); do echo "--- gdb $p $(tr '\0' ' ' < /proc/$p/cmdline | cut -c1-120)"; timeout 120 gdb -p "$p" -batch -ex "thread apply all bt 25" 2>&1 | grep -E "^#|^Thread" | head -300; done
    for f in /tmp/nccl.*.log; do echo "--- $f"; tail -15 "$f"; done
    kill -9 "$PID"; break
  fi
done
wait "$PID" || true
echo "=== pytest output"; grep -vE "PASSED|SKIPPED" /tmp/pytest.out | tail -200
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
