#!/bin/bash

# Copyright (c) 2019-2024, NVIDIA CORPORATION.

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
   cugraph-dgl
   docs
   all
   -v
   -g
   -n
   --pydevelop
   --allgpuarch
   --clean
   -h
   --help
"

HELP="$0 [<target> ...] [<flag> ...]
 where <target> is:
   clean                      - remove all existing build artifacts and configuration (start over)
   uninstall                  - uninstall libcugraph and cugraph from a prior build/install (see also -n)
   cugraph-pyg                - build the cugraph-pyg Python package
   cugraph-dgl                - build the cugraph-dgl extensions for DGL
   docs                       - build the docs
   all                        - build everything
 and <flag> is:
   -v                         - verbose build mode
   -g                         - build for debug
   -n                         - do not install after a successful build (does not affect Python packages)
   --pydevelop                - install the Python packages in editable mode
   --allgpuarch               - build for all supported GPU architectures
   --clean                    - clean an individual target (note: to do a complete rebuild, use the clean target described above)
   -h                         - print this text

 default action (no args) is to build and install 'cugraph-pyg' then 'cugraph-dgl' then 'wholegraph' targets

"

CUGRAPH_PYG_BUILD_DIR=${REPODIR}/python/cugraph-pyg/build
CUGRAPH_DGL_BUILD_DIR=${REPODIR}/python/cugraph-dgl/build

BUILD_DIRS="${CUGRAPH_PYG_BUILD_DIR}
            ${CUGRAPH_DGL_BUILD_DIR}
"

# Set defaults for vars modified by flags to this script
VERBOSE_FLAG=""
BUILD_TYPE=Release
INSTALL_TARGET="--target install"
BUILD_ALL_GPU_ARCH=0
PYTHON_ARGS_FOR_INSTALL="-m pip install --no-build-isolation --no-deps"

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
    PYTHON_ARGS_FOR_INSTALL="${PYTHON_ARGS_FOR_INSTALL} -e"
fi

# If clean or uninstall targets given, run them prior to any other steps
if hasArg uninstall; then
    # uninstall cugraph and pylibcugraph installed from a prior install
    # FIXME: if multiple versions of these packages are installed, this only
    # removes the latest one and leaves the others installed. build.sh uninstall
    # can be run multiple times to remove all of them, but that is not obvious.
    pip uninstall -y  cugraph-dgl cugraph-pyg
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
# Build and install the cugraph-pyg Python package
if hasArg cugraph-pyg || buildDefault || hasArg all; then
    if hasArg --clean; then
        cleanPythonDir ${REPODIR}/python/cugraph-pyg
    else
        python ${PYTHON_ARGS_FOR_INSTALL} ${REPODIR}/python/cugraph-pyg
    fi
fi

# Install the cugraph-dgl extensions for DGL
if hasArg cugraph-dgl || buildDefault ||hasArg all; then
    if hasArg --clean; then
        cleanPythonDir ${REPODIR}/python/cugraph-dgl
    else
        python ${PYTHON_ARGS_FOR_INSTALL} ${REPODIR}/python/cugraph-dgl
    fi
fi

# Build the docs
if hasArg docs || hasArg all; then
    if [ ! -d ${LIBCUGRAPH_BUILD_DIR} ]; then
        mkdir -p ${LIBCUGRAPH_BUILD_DIR}
        cd ${LIBCUGRAPH_BUILD_DIR}
        cmake -B "${LIBCUGRAPH_BUILD_DIR}" -S "${REPODIR}/cpp" \
              -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
              -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
              ${CMAKE_GENERATOR_OPTION} \
              ${CMAKE_VERBOSE_OPTION}
    fi

    for PROJECT in libcugraphops libwholegraph; do
        XML_DIR="${REPODIR}/docs/cugraph/${PROJECT}"
        rm -rf "${XML_DIR}"
        mkdir -p "${XML_DIR}"
        export XML_DIR_${PROJECT^^}="$XML_DIR"

        echo "downloading xml for ${PROJECT} into ${XML_DIR}. Environment variable XML_DIR_${PROJECT^^} is set to ${XML_DIR}"
        curl -O "https://d1664dvumjb44w.cloudfront.net/${PROJECT}/xml_tar/${RAPIDS_VERSION}/xml.tar.gz"
        tar -xzf xml.tar.gz -C "${XML_DIR}"
        rm "./xml.tar.gz"
    done

    cd ${LIBCUGRAPH_BUILD_DIR}
    cmake --build "${LIBCUGRAPH_BUILD_DIR}" -j${PARALLEL_LEVEL} --target docs_cugraph ${VERBOSE_FLAG}

    echo "making libcugraph doc dir"
    rm -rf ${REPODIR}/docs/cugraph/libcugraph
    mkdir -p ${REPODIR}/docs/cugraph/libcugraph

    export XML_DIR_LIBCUGRAPH="${REPODIR}/cpp/doxygen/xml"

    cd ${REPODIR}/docs/cugraph
    make html
fi
