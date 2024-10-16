#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

version=$(rapids-generate-version)
git_commit=$(git rev-parse HEAD)
export RAPIDS_PACKAGE_VERSION=${version}
echo "${version}" > VERSION

rapids-logger "Begin py build"

package_dir="python"
for package_name in cugraph-pyg cugraph-dgl; do
  underscore_package_name=$(echo "${package_name}" | tr "-" "_")
  sed -i "/^__git_commit__/ s/= .*/= \"${git_commit}\"/g" "${package_dir}/${package_name}/${underscore_package_name}/_version.py"
done

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"

if [[ ${RAPIDS_CUDA_MAJOR} == "11" ]]; then
  # Only CUDA 11 is supported right now due to PyTorch requirement.
  rapids-conda-retry mambabuild \
    --no-test \
    --channel "${CPP_CHANNEL}" \
    --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
    --channel pyg \
    --channel pytorch \
    --channel pytorch-nightly \
    conda/recipes/cugraph-pyg

  # Only CUDA 11 is supported right now due to PyTorch requirement.
  rapids-conda-retry mambabuild \
    --no-test \
    --channel "${CPP_CHANNEL}" \
    --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
    --channel dglteam \
    --channel pytorch \
    --channel pytorch-nightly \
    conda/recipes/cugraph-dgl
fi

rapids-upload-conda-to-s3 python
