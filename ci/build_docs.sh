#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Downloading artifacts from previous jobs"

CPP_CHANNEL=$(rapids-download-conda-from-github cpp)

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

export RAPIDS_VERSION_MAJOR_MINOR="$(rapids-version-major-minor)"

rapids-dependency-file-generator \
  --output conda \
  --file-key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  --prepend-channel "${CPP_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n docs

# Temporarily allow unbound variables for conda activation.
set +u
conda activate docs
set -u

rapids-print-env

export RAPIDS_DOCS_DIR="$(mktemp -d)"

rapids-logger "Build C++ docs"
pushd cpp
doxygen Doxyfile
mkdir -p "${RAPIDS_DOCS_DIR}/libwholegraph/xml_tar"
tar -czf "${RAPIDS_DOCS_DIR}/libwholegraph/xml_tar"/xml.tar.gz -C xml .
popd

rapids-logger "Output temp dir: ${RAPIDS_DOCS_DIR}"

RAPIDS_VERSION_NUMBER="$(rapids-version-major-minor)" rapids-upload-docs
