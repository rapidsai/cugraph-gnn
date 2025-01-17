# Copyright (c) 2025, NVIDIA CORPORATION.

CUGRAPH_COMMIT="2dbd946375d83dc30d29f1b09a013e3a6159483d"

CUGRAPH_CPP_CHANNEL=$(rapids-get-pr-conda-artifact cugraph 4804 cpp "${CUGRAPH_COMMIT:0:7}")
CUGRAPH_PYTHON_CHANNEL=$(rapids-get-pr-conda-artifact cugraph 4804 python "${CUGRAPH_COMMIT:0:7}")

conda config --system --add channels "${CUGRAPH_CPP_CHANNEL}"
conda config --system --add channels "${CUGRAPH_PYTHON_CHANNEL}"
