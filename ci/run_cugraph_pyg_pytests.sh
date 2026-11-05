#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking run_cugraph_pyg_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cugraph-pyg/cugraph_pyg

export PYTHONUNBUFFERED=1
df -h /dev/shm; ip -br addr || true
echo "=== V1 single file"; NCCL_DEBUG=WARN timeout 600 pytest -v -rs -p no:cacheprovider -o faulthandler_timeout=240 tests/data/test_feature_store.py -k basic_api 2>&1 | grep -E "PASS|FAIL|SKIP|Timeout|Fatal|WARN" | head -20 || true
echo "=== V2 full collection"; NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,ENV,SHM,PROXY,NET timeout 600 pytest -v -rs -p no:cacheprovider -o faulthandler_timeout=240 . -k test_feature_store_basic_api 2>&1 | grep -vE "NCCL INFO Channel|Connected all|NCCL INFO P2P" | tail -250 || true
echo "=== V3 data+loader"; NCCL_DEBUG=WARN timeout 600 pytest -v -rs -p no:cacheprovider -o faulthandler_timeout=240 tests/data tests/loader -k test_feature_store_basic_api 2>&1 | grep -E "PASS|FAIL|SKIP|Timeout|Fatal|WARN" | head -20 || true
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
