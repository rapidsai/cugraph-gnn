# Copyright (c) 2025, NVIDIA CORPORATION.

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

CUGRAPH_COMMIT="12a01e2cb48cd6018bf9a2332a4975825307fea7"
CUGRAPH_CHANNEL=$(
  RAPIDS_PY_WHEEL_NAME="cugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact cugraph 4804 python "${CUGRAPH_COMMIT:0:}"
)
LIBCUGRAPH_CHANNEL=$(
  RAPIDS_PY_WHEEL_NAME="libcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact cugraph 4804 cpp "${CUGRAPH_COMMIT:0:7}"
)
PYLIBCUGRAPH_CHANNEL=$(
  RAPIDS_PY_WHEEL_NAME="pylibcugraph_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact cugraph 4804 python "${CUGRAPH_COMMIT:0:}"
)

cat > ./constraints.txt <<EOF
cugraph-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${CUGRAPH_CHANNEL}/cugraph_*.whl)
libcugraph-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBCUGRAPH_CHANNEL}/libcugraph_*.whl)
pylibcugraph-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${PYLIBCUGRAPH_CHANNEL}/pylibcugraph_*.whl)
EOF

export PIP_CONSTRAINT=$(pwd)/constraints.txt
