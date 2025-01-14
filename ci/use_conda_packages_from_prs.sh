# Copyright (c) 2025, NVIDIA CORPORATION.

RAFT_COMMIT="4b793be27b27d40119706ea5df26cc03c8efe33c"

RAFT_CPP_CHANNEL=$(rapids-get-pr-conda-artifact raft 2531 cpp "${RAFT_COMMIT:0:7}")
RAFT_PYTHON_CHANNEL=$(rapids-get-pr-conda-artifact raft 2531 python "${RAFT_COMMIT:0:7}")

CUGRAPH_COMMIT="12a01e2cb48cd6018bf9a2332a4975825307fea7"

CUGRAPH_CPP_CHANNEL=$(rapids-get-pr-conda-artifact cugraph 4804 cpp "${CUGRAPH_COMMIT:0:7}")
CUGRAPH_PYTHON_CHANNEL=$(rapids-get-pr-conda-artifact cugraph 4804 python "${CUGRAPH_COMMIT:0:7}")

conda config --system --add channels "${RAFT_CPP_CHANNEL}"
conda config --system --add channels "${RAFT_PYTHON_CHANNEL}"

conda config --system --add channels "${CUGRAPH_CPP_CHANNEL}"
conda config --system --add channels "${CUGRAPH_PYTHON_CHANNEL}"
