#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e -u -o pipefail

# Ensure CUDA-enabled 'torch' packages are always used.
#
# Downloading + adding the downloaded file to the constraint forces the use of this
# package, so we don't accidentally end up with a CPU-only 'torch' from 'pypi.org'
# (which can happen because --extra-index-url doesn't imply a priority).
rapids-logger "Downloading 'torch' wheel"
CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
if [[ "${CUDA_MAJOR}" == "12" ]]; then
  PYTORCH_INDEX="https://download.pytorch.org/whl/cu126"
else
  PYTORCH_INDEX="https://download.pytorch.org/whl/cu130"
fi

TORCH_WHEEL_DIR=$(mktemp -d)
pip download \
  --prefer-binary \
  --no-dps \
  --constraint "${PIP_CONSTRAINT}" \
  --index-url "${PYTORCH_INDEX}" \
  'torch'

echo "torch @ file://$(echo ${TORCH_WHEEL_DIR}/torch_*.whl)" >> "${PIP_CONSTRAINT}"
