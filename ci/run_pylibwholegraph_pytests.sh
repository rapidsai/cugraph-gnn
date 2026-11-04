#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if grep -qi microsoft /proc/version; then export NCCL_CUMEM_ENABLE=0; fi

# Support invoking run_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/pylibwholegraph/pylibwholegraph/

pytest -rs --cache-clear --import-mode=append "$@" tests
