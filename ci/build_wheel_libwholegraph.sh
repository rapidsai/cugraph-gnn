#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

package_name="libwholegraph"
package_dir="python/libwholegraph"

export SKBUILD_CMAKE_ARGS="-DBUILD_SHARED_LIBS=OFF;-DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE;-DCUDA_STATIC_RUNTIME=ON"

EXCLUDE_ARGS=(
    --exclude libcuda.so.1
    --exclude libnvidia-ml.so.1
    --exclude librapids_logger.so
)

source rapids-configure-sccache
source rapids-date-string

rapids-generate-version > ./VERSION

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

cd "${package_dir}"

sccache --zero-stats

rapids-logger "Building '${package_name}' wheel"
rapids-pip-retry wheel \
    -w dist \
    -v \
    --no-deps \
    --disable-pip-version-check \
    .

sccache --show-adv-stats

python -m auditwheel repair \
    "${EXCLUDE_ARGS[@]}" \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    dist/*

./ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 cpp "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
