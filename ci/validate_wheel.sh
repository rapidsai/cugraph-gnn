#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

package_dir=$1
wheel_dir_relative_path=$2

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"

python -m pip install \
    --prefer-binary \
    'pkginfo>=1.12.1.2'

cd "${package_dir}"

rapids-logger "validate packages with 'pydistcheck'"

PYDISTCHECK_ARGS=(
    --inspect
)

# PyPI hard limit is 1GiB, but try to keep these as small as possible
if [[ "${package_dir}" == "python/libwholegraph" ]]; then
    if [[ "${RAPIDS_CUDA_MAJOR}" == "12" ]]; then
        PYDISTCHECK_ARGS+=(
            --max-allowed-size-compressed '75Mi'
        )
    else
        PYDISTCHECK_ARGS+=(
            --max-allowed-size-compressed '40Mi'
        )
    fi
elif [[ "${package_dir}" != "python/cugraph-pyg" ]] && \
     [[ "${package_dir}" != "python/pylibwholegraph" ]]; then
    rapids-echo-stderr "unrecognized package_dir: '${package_dir}'"
    exit 1
fi

pydistcheck \
    "${PYDISTCHECK_ARGS[@]}" \
    "$(echo ${wheel_dir_relative_path}/*.whl)"

rapids-logger "validate packages with 'twine'"

twine check \
    --strict \
    "$(echo ${wheel_dir_relative_path}/*.whl)"

rapids-logger "validating that the wheel doesn't depend on 'torch' (even in an extra)"
WHEEL_FILE="$(${wheel_dir_relative_path}/*.whl)"

# NOTE: group of specifiers after 'torch' to avoid a false positive like 'torch-geometric'
unzip -p "${WHEEL_FILE}" '*.dist-info/METADATA' \
| grep -E '^Requires-Dist:.*torch[><=!~ ]+.*' \
| tee matches.txt

if wc -l < ./matches.txt; then
    echo -n "Wheel '${WHEEL_FILE}' appears to depend on 'torch'. Remove that dependency. "
    echo -n "We prefer to not declare a 'torch' dependency and allow it to be managed separately, "
    echo "to ensure tight control over the variants installed (including for DLFW builds)."
    exit 1
else
    echo "No dependency on 'torch' found"
    exit 0
fi
