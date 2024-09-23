#!/bin/bash
# Copyright (c) 2018-2024, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

## Usage
# bash update-version.sh <new_version>


# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')
CURRENT_MAJOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[1]}')
CURRENT_MINOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[2]}')
CURRENT_PATCH=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[3]}')
CURRENT_SHORT_TAG=${CURRENT_MAJOR}.${CURRENT_MINOR}

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}
NEXT_UCX_PY_VERSION="$(curl -sL https://version.gpuci.io/rapids/${NEXT_SHORT_TAG})"

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

# Centralized version file update
# NOTE: Any script that runs in CI will need to use gha-tool `rapids-generate-version`
# and echo it to `VERSION` file to get an alpha spec of the current version
echo "${NEXT_FULL_TAG}" > VERSION

# Need to distutils-normalize the original version
NEXT_SHORT_TAG_PEP440=$(python -c "from setuptools.extern import packaging; print(packaging.version.Version('${NEXT_SHORT_TAG}'))")

DEPENDENCIES=(
  cudf
  cugraph
  cugraph-dgl
  cugraph-pyg
  cugraph-service-server
  cugraph-service-client
  cuxfilter
  dask-cuda
  dask-cudf
  libcudf
  libcugraphops
  libraft
  libraft-headers
  librmm
  pylibcugraph
  pylibcugraphops
  pylibwholegraph
  pylibraft
  pyraft
  raft-dask
  rmm
  ucx-py
  rapids-dask-dependency
)
for DEP in "${DEPENDENCIES[@]}"; do
  for FILE in dependencies.yaml conda/environments/*.yaml python/cugraph-{pyg,dgl}/conda/*.yaml; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*/g" "${FILE}"
    sed_runner "/-.* ucx-py==/ s/==.*/==${NEXT_UCX_PY_VERSION}.*/g" "${FILE}"
  done
  for FILE in python/**/pyproject.toml; do
    sed_runner "/\"${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*\"/==${NEXT_SHORT_TAG_PEP440}.*\"/g" "${FILE}"
    sed_runner "/\"ucx-py==/ s/==.*\"/==${NEXT_UCX_PY_VERSION}.*\"/g" "${FILE}"
  done
done

sed_runner "s/\(PROJECT_NUMBER[[:space:]]*\)=.*/\1= ${NEXT_SHORT_TAG}/" cpp/Doxyfile
sed_runner "s/set(RAPIDS_VERSION *\"[0-9.]*\")/set(RAPIDS_VERSION \"${NEXT_SHORT_TAG}\")/" cpp/CMakeLists.txt
sed_runner "s/set(RAPIDS_VERSION *\"[0-9.]*\")/set(RAPIDS_VERSION \"${NEXT_SHORT_TAG}\")/" python/pylibwholegraph/CMakeLists.txt
