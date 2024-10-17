#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_name=$1
package_dir=$2
underscore_package_name=$(echo "${package_name}" | tr "-" "_")

# The set of shared libraries that should be packaged differs by project.
#
# Capturing that here in argument-parsing to allow this build_wheel.sh
# script to be re-used by all wheel builds in the project.
case "${package_dir}" in
  python/pylibwholegraph)
    EXCLUDE_ARGS=(
        --exclude libcuda.so.1
        --exclude libnvidia-ml.so.1
    )
  ;;
  *)
    EXCLUDE_ARGS=()
  ;;
esac

source rapids-configure-sccache
source rapids-date-string

rapids-generate-version > ./VERSION

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

cd "${package_dir}"

rapids-logger "Building '${package_name}' wheel"
python -m pip wheel \
    -w dist \
    -v \
    --no-deps \
    --disable-pip-version-check \
    .

sccache --show-adv-stats

# pure-python packages should be marked as pure, and not have auditwheel run on them.
if [[ ${package_name} == "cugraph-dgl" ]] || \
   [[ ${package_name} == "cugraph-pyg" ]]; then
    RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-upload-wheels-to-s3 python dist
else

    mkdir -p final_dist
    python -m auditwheel repair \
        "${EXCLUDE_ARGS[@]}" \
        -w final_dist \
        dist/*

    RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 final_dist
fi
