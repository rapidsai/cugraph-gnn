#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <n> [pytest-args...]"
  echo "Runs the disjoint sampling neighbor loader tests <n> times."
  exit 1
fi

n="$1"
shift

if ! [[ "$n" =~ ^[0-9]+$ ]] || [ "$n" -le 0 ]; then
  echo "Error: run count must be a positive integer."
  exit 1
fi

cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/..

for i in $(seq 1 "$n"); do
  echo "============================================================"
  echo "Run ${i}/${n}: pytest -v -s -x -k 'disjoint' python/cugraph-pyg/cugraph_pyg/tests/loader/test_neighbor_loader.py $*"
  echo "============================================================"
  pytest -v -s -x -k "disjoint" python/cugraph-pyg/cugraph_pyg/tests/loader/test_neighbor_loader.py "$@"
  echo "Completed run ${i}/${n}."
  echo
 done
