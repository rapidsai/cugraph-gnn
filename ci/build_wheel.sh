#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

package_name=$1
package_dir=$2
package_type=$3

# The set of shared libraries that should be packaged differs by project.
#
# Capturing that here in argument-parsing to allow this build_wheel.sh
# script to be re-used by all wheel builds in the project.
#
EXCLUDE_ARGS=(
    --exclude libcuda.so.1
    --exclude "libnccl.so.*"
    --exclude libnvidia-ml.so.1
    --exclude librapids_logger.so
    --exclude librmm.so
)

if [[ "${package_name}" != "libwholegraph" ]]; then
    EXCLUDE_ARGS+=(
        --exclude libwholegraph.so
    )
fi

source rapids-configure-sccache
source rapids-date-string

export SCCACHE_S3_PREPROCESSOR_CACHE_KEY_PREFIX="${package_name}/${RAPIDS_CONDA_ARCH}/cuda${RAPIDS_CUDA_VERSION%%.*}/wheel/preprocessor-cache"
export SCCACHE_S3_USE_PREPROCESSOR_CACHE_MODE=true

rapids-generate-version > ./VERSION

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

cd "${package_dir}"

sccache --stop-server 2>/dev/null || true

rapids-logger "Building '${package_name}' wheel"

RAPIDS_PIP_WHEEL_ARGS=(
  -w dist
  -v
  --no-deps
  --disable-pip-version-check
  --build-constraint="${PIP_CONSTRAINT}"
)

# unset PIP_CONSTRAINT (set by rapids-init-pip)... it doesn't affect builds as of pip 25.3, and
# results in an error from 'pip wheel' when set and --build-constraint is also passed
unset PIP_CONSTRAINT
rapids-pip-retry wheel \
  "${RAPIDS_PIP_WHEEL_ARGS[@]}" \
  .

sccache --show-adv-stats
sccache --stop-server >/dev/null 2>&1 || true

# pure-python packages should be marked as pure, and not have auditwheel run on them.
if [[ ${package_name} == "cugraph-pyg" ]]; then
    cp dist/* "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}/"
else
    # repair wheels and write to the location that artifact-uploading code expects to find them
    python -m auditwheel repair \
        "${EXCLUDE_ARGS[@]}" \
        -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
        dist/*
fi
