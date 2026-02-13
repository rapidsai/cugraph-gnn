#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

package_name=$1
package_dir=$2
shift 2

# Parse optional flags
stable_abi=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stable)
      stable_abi=true
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

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
source rapids-init-pip

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
)

# Add py-api setting for stable ABI builds
if [[ "${stable_abi}" == "true" ]] && [[ -n "${RAPIDS_PY_API:-}" ]]; then
  RAPIDS_PIP_WHEEL_ARGS+=(--config-settings="skbuild.wheel.py-api=${RAPIDS_PY_API}")
fi

# Only use --build-constraint when build isolation is enabled.
#
# Passing '--build-constraint' and '--no-build-isolation` together results in an error from 'pip',
# but we want to keep environment variable PIP_CONSTRAINT set unconditionally.
# PIP_NO_BUILD_ISOLATION=0 means "add --no-build-isolation" (ref: https://github.com/pypa/pip/issues/5735)
if [[ "${PIP_NO_BUILD_ISOLATION:-}" != "0" ]]; then
    RAPIDS_PIP_WHEEL_ARGS+=(--build-constraint="${PIP_CONSTRAINT}")
fi

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
