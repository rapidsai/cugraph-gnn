#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# [description]
#
#   Downloads a CUDA variant of 'torch' from the correct index, based on CUDA major version.
#
#   This exists to avoid using 'pip --extra-index-url', which has these undesirable properties:
#
#     - allows for CPU-only 'torch' to be downloaded from pypi.org
#     - allows for other non-torch packages like 'numpy' to be downloaded from the PyTorch indices
#     - increases solve complexity for 'pip'
#

set -e -u -o pipefail

TORCH_WHEEL_DIR="${1}"

# skip download attempt on CUDA versions where we know there isn't a 'torch' CUDA wheel.
CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
CUDA_MINOR=$(echo "${RAPIDS_CUDA_VERSION}" | cut -d'.' -f2)
if \
    { [ "${CUDA_MAJOR}" -eq 12 ] && [ "${CUDA_MINOR}" -lt 9 ]; } \
    || { [ "${CUDA_MAJOR}" -eq 13 ] && [ "${CUDA_MINOR}" -gt 0 ]; } \
    || [ "${CUDA_MAJOR}" -gt 13 ];
then
    rapids-logger "Skipping 'torch' wheel download. (requires CUDA 12.9+ or 13.0, found ${RAPIDS_CUDA_VERSION})"
    exit 0
fi

# Ensure CUDA-enabled 'torch' packages are always used.
#
# Downloading + passing the downloaded file as a requirement forces the use of this
# package and ensures 'pip' considers all of its requirements.
#
# Not appending this to PIP_CONSTRAINT, because we don't want the torch '--extra-index-url'
# to leak outside of this script into other 'pip {download,install}'' calls.
rapids-dependency-file-generator \
    --output requirements \
    --file-key "torch_only" \
    --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES};require_gpu=true" \
| tee ./torch-constraints.txt

rapids-pip-retry download \
  --isolated \
  --prefer-binary \
  --no-deps \
  -d "${TORCH_WHEEL_DIR}" \
  --constraint "${PIP_CONSTRAINT}" \
  --constraint ./torch-constraints.txt \
  'torch'
