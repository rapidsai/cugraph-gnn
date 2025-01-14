# Copyright (c) 2025, NVIDIA CORPORATION.

RAFT_COMMIT="d275c995fb51310d1340fe2fd6d63d0bfd43cafa"

RAFT_CPP_CHANNEL=$(rapids-get-pr-conda-artifact raft 2531 cpp "${RAFT_COMMIT:0:7}")
RAFT_PYTHON_CHANNEL=$(rapids-get-pr-conda-artifact raft 2531 python "${RAFT_COMMIT:0:7}")

CUGRAPH_COMMIT="8fe1d33cbcaf1f40a6b3d06ec48cc699c47f8b44"

CUGRAPH_CPP_CHANNEL=$(rapids-get-pr-conda-artifact cugraph 4804 cpp "${CUGRAPH_COMMIT:0:7}")
CUGRAPH_PYTHON_CHANNEL=$(rapids-get-pr-conda-artifact cugraph 4804 python "${CUGRAPH_COMMIT:0:7}")

conda config --system --add channels "${RAFT_CPP_CHANNEL}"
conda config --system --add channels "${RAFT_PYTHON_CHANNEL}"

conda config --system --add channels "${CUGRAPH_CPP_CHANNEL}"
conda config --system --add channels "${CUGRAPH_PYTHON_CHANNEL}"
