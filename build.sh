#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# cugraph build script

# This script is used to build the component(s) in this repo from
# source, and can be called with various options to customize the
# build as needed (see the help output for details)

# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

# NOTE: ensure all dir changes are relative to the location of this
# script, and that this script resides in the repo dir!
REPODIR=$(cd $(dirname $0); pwd)

RAPIDS_VERSION="$(sed -E -e 's/^([0-9]{2})\.([0-9]{2})\.([0-9]{2}).*$/\1.\2/' VERSION)"

# Valid args to this script (all possible targets and options) - only one per line
VALIDARGS="
   clean
   uninstall
   cugraph-pyg
   pylibwholegraph
   libwholegraph
   tests
   benchmarks
   all
   -v
   -g
   -n
   --pydevelop
   --allgpuarch
   --compile-cmd
   --clean
   -h
   --help
"

HELP="$0 [<target> ...] [<flag> ...]
 where <target> is:
   clean                      - remove all existing build artifacts and configuration (start over)
   uninstall                  - uninstall libwholegraph and GNN Python packages from a prior build/install (see also -n)
   cugraph-pyg                - build the cugraph-pyg Python package
   pylibwholegraph            - build the pylibwholegraph Python package
   libwholegraph              - build the libwholegraph library
   tests                      - build the C++ tests
   benchmarks                 - build benchmarks
   all                        - build everything
 and <flag> is:
   -v                         - verbose build mode
   -g                         - build for debug
   -n                         - do not install after a successful build (does not affect Python packages)
   --pydevelop                - install the Python packages in editable mode
   --allgpuarch               - build for all supported GPU architectures
   --enable-nvshmem            - build with nvshmem support (beta).
   --compile-cmd               - only output compile commands (invoke CMake without build)
   --clean                    - clean an individual target (note: to do a complete rebuild, use the clean target described above)
   -h                         - print this text

 default action (no args) is to build and install 'libwholegraph' then 'pylibwholegraph' then 'cugraph-pyg'

"

CUGRAPH_PYG_BUILD_DIR=${REPODIR}/python/cugraph-pyg/build
PYLIBWHOLEGRAPH_BUILD_DIR=${REPODIR}/python/pylibwholegraph/build
LIBWHOLEGRAPH_BUILD_DIR=${REPODIR}/cpp/build

BUILD_DIRS="${CUGRAPH_PYG_BUILD_DIR}
            ${PYLIBWHOLEGRAPH_BUILD_DIR}
            ${LIBWHOLEGRAPH_BUILD_DIR}
"

# Set defaults for vars modified by flags to this script
VERBOSE_FLAG=""
BUILD_TYPE=Release
INSTALL_TARGET="--target install"
BUILD_ALL_GPU_ARCH=0
PYTHON_ARGS_FOR_INSTALL=(
    --no-build-isolation
    --no-deps
    --config-settings="rapidsai.disable-cuda=true"
)

# Set defaults for vars that may not have been defined externally
#  FIXME: if PREFIX is not set, check CONDA_PREFIX, but there is no fallback
#  from there!
INSTALL_PREFIX=${PREFIX:=${CONDA_PREFIX}}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=`nproc`}
BUILD_ABI=${BUILD_ABI:=ON}

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

function buildDefault {
    (( ${NUMARGS} == 0 )) || !(echo " ${ARGS} " | grep -q " [^-][a-zA-Z0-9\_\-]\+ ")
}

function cleanPythonDir {
    pushd $1 > /dev/null
    find . -type d -name __pycache__ -print | xargs rm -rf
    find . -type d -name build -print | xargs rm -rf
    find . -type d -name dist -print | xargs rm -rf
    find . -type f -name "*.cpp" -delete
    find . -type f -name "*.cpython*.so" -delete
    find . -type d -name _external_repositories -print | xargs rm -rf
    popd > /dev/null
}

if hasArg -h || hasArg --help; then
    echo "${HELP}"
    exit 0
fi

# Check for valid usage
if (( ${NUMARGS} != 0 )); then
    for a in ${ARGS}; do
        if ! (echo "${VALIDARGS}" | grep -q "^[[:blank:]]*${a}$"); then
            echo "Invalid option: ${a}"
            exit 1
        fi
    done
fi

# Process flags
if hasArg -v; then
    VERBOSE_FLAG="-v"
    CMAKE_VERBOSE_OPTION="--log-level=VERBOSE"
fi
if hasArg -g; then
    BUILD_TYPE=Debug
fi
if hasArg -n; then
    INSTALL_TARGET=""
fi
if hasArg --allgpuarch; then
    BUILD_ALL_GPU_ARCH=1
fi
if hasArg --pydevelop; then
    PYTHON_ARGS_FOR_INSTALL+=("-e")
fi

if hasArg --enable-nvshmem; then
    BUILD_WITH_NVSHMEM=ON
else
    BUILD_WITH_NVSHMEM=OFF
fi
if hasArg tests; then
    BUILD_TESTS=ON
else
    BUILD_TESTS=OFF
fi
if hasArg benchmarks; then
    BUILD_BENCHMARKS=ON
else
    BUILD_BENCHMARKS=OFF
fi


# If clean or uninstall targets given, run them prior to any other steps
if hasArg uninstall; then
    if [[ "$INSTALL_PREFIX" != "" ]]; then
        rm -rf ${INSTALL_PREFIX}/include/wholememory
        rm -f ${INSTALL_PREFIX}/lib/libwholegraph.so
        rm -rf ${INSTALL_PREFIX}/lib/cmake/wholegraph
    fi
    # This may be redundant given the above, but can also be used in case
    # there are other installed files outside of the locations above.
    if [ -e ${LIBWHOLEGRAPH_BUILD_DIR}/install_manifest.txt ]; then
        xargs rm -f < ${LIBWHOLEGRAPH_BUILD_DIR}/install_manifest.txt > /dev/null 2>&1
    fi

    # uninstall cugraph-pyg/wholegraph installed from a prior install
    # FIXME: if multiple versions of these packages are installed, this only
    # removes the latest one and leaves the others installed. build.sh uninstall
    # can be run multiple times to remove all of them, but that is not obvious.
    pip uninstall -y  cugraph-pyg pylibwholegraph libwholegraph
fi

if hasArg clean; then
    # Ignore errors for clean since missing files, etc. are not failures
    set +e
    # remove artifacts generated inplace
    if [[ -d ${REPODIR}/python ]]; then
        cleanPythonDir ${REPODIR}/python
    fi

    # If the dirs to clean are mounted dirs in a container, the contents should
    # be removed but the mounted dirs will remain.  The find removes all
    # contents but leaves the dirs, the rmdir attempts to remove the dirs but
    # can fail safely.
    for bd in ${BUILD_DIRS}; do
        if [ -d ${bd} ]; then
            find ${bd} -mindepth 1 -delete
            rmdir ${bd} || true
        fi
    done
    # Go back to failing on first error for all other operations
    set -e
fi

################################################################################
# Build and install the libwholegraph library
if hasArg libwholegraph || buildDefault || hasArg all ; then

    # set values based on flags
    if (( BUILD_ALL_GPU_ARCH == 0 )); then
        WHOLEGRAPH_CMAKE_CUDA_ARCHITECTURES="NATIVE"
        echo "Building for the architecture of the GPU in the system..."
    else
        WHOLEGRAPH_CMAKE_CUDA_ARCHITECTURES="RAPIDS"
        echo "Building for *ALL* supported GPU architectures..."
    fi

    cmake -S ${REPODIR}/cpp -B ${LIBWHOLEGRAPH_BUILD_DIR} \
          -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
          -DCMAKE_CUDA_ARCHITECTURES=${WHOLEGRAPH_CMAKE_CUDA_ARCHITECTURES} \
          -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
          -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} \
          -DCMAKE_MESSAGE_LOG_LEVEL=VERBOSE \
          -DBUILD_TESTS=${BUILD_TESTS} \
          -DBUILD_WITH_NVSHMEM=${BUILD_WITH_NVSHMEM}

    cd ${LIBWHOLEGRAPH_BUILD_DIR}

    if ! hasArg --compile-cmd; then
        ## Build and (optionally) install library + tests
        cmake --build . -j${PARALLEL_LEVEL} ${INSTALL_TARGET} ${VERBOSE_FLAG}
    fi
fi

# Build and install the pylibwholegraph Python package
if hasArg pylibwholegraph || buildDefault || hasArg all; then
    if hasArg --clean; then
        cleanPythonDir ${REPODIR}/python/pylibwholegraph
    fi

    # If `RAPIDS_PY_VERSION` is set, use that as the lower-bound for the stable ABI CPython version
    if [ -n "${RAPIDS_PY_VERSION:-}" ]; then
        RAPIDS_PY_API="cp${RAPIDS_PY_VERSION//./}"
        PYTHON_ARGS_FOR_INSTALL+=("--config-settings" "skbuild.wheel.py-api=${RAPIDS_PY_API}")
    fi

    # setup.py and cmake reference an env var LIBWHOLEGRAPH_DIR to find the
    # libwholegraph package (cmake).
    # If not set by the user, set it to LIBWHOLEGRAPH_BUILD_DIR
    LIBWHOLEGRAPH_DIR=${LIBWHOLEGRAPH_DIR:=${LIBWHOLEGRAPH_BUILD_DIR}}
    if ! hasArg --compile-cmd; then
        cd ${REPODIR}/python/pylibwholegraph
        env LIBWHOLEGRAPH_DIR=${LIBWHOLEGRAPH_DIR} \
        SKBUILD_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=${BUILD_TYPE}" python -m pip install \
            "${PYTHON_ARGS_FOR_INSTALL[@]}" \
            .
    else
        # just invoke cmake without going through scikit-build-core
        env LIBWHOLEGRAPH_DIR=${LIBWHOLEGRAPH_DIR} \
        cmake -S ${REPODIR}/python/pylibwholegraph -B ${REPODIR}/python/pylibwholegraph/build \
           -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
    fi
fi

# Build and install the cugraph-pyg Python package
if hasArg cugraph-pyg || buildDefault || hasArg all; then
    if hasArg --clean; then
        cleanPythonDir ${REPODIR}/python/cugraph-pyg
    else
        python -m pip install \
            "${PYTHON_ARGS_FOR_INSTALL[@]}" \
            "${REPODIR}/python/cugraph-pyg"
    fi
fi
