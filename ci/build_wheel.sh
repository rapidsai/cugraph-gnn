#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_name=$1
package_dir=$2
underscore_package_name=$(echo "${package_name}" | tr "-" "_")

source rapids-configure-sccache
source rapids-date-string

rapids-generate-version > ./VERSION

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

cd "${package_dir}"

WHEEL_BUILD_ARGS="-m pip wheel . -w dist -v --no-deps --disable-pip-version-check"

# pure-python packages should be marked as pure, and not have auditwheel run on them.
if [[ ${package_name} == "cugraph-dgl" ]] || \
   [[ ${package_name} == "cugraph-pyg" ]]; then
    python ${WHEEL_BUILD_ARGS}
    RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-upload-wheels-to-s3 python dist
else
    # Hardcode the output dir
    SKBUILD_CMAKE_ARGS="-DDETECT_CONDA_ENV=OFF;-DBUILD_SHARED_LIBS=OFF;-DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE;-DCUDA_STATIC_RUNTIME=ON;-DWHOLEGRAPH_BUILD_WHEELS=ON" \
    python ${WHEEL_BUILD_ARGS}

    mkdir -p final_dist
    python -m auditwheel repair \
        --exclude libcuda.so.1 \
        --exclude libnvidia-ml.so.1 \
        -w final_dist \
        dist/*

    RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 final_dist
fi
