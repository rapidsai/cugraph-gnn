# Copyright (c) 2025, NVIDIA CORPORATION.

CUGRAPH_COMMIT="5b6fd89c2e44ef58c22131aa799807151bff4e42"

CUGRAPH_CPP_CHANNEL=$(rapids-get-pr-conda-artifact cugraph 4804 cpp "${CUGRAPH_COMMIT:0:7}")
CUGRAPH_PYTHON_CHANNEL=$(rapids-get-pr-conda-artifact cugraph 4804 python "${CUGRAPH_COMMIT:0:7}")

conda config --system --add channels "${CUGRAPH_CPP_CHANNEL}"
conda config --system --add channels "${CUGRAPH_PYTHON_CHANNEL}"
