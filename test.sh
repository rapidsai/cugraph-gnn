

docker run \
    --rm \
    --gpus all \
    --env GH_TOKEN=$(gh auth token) \
    -v $(pwd):/opt/work \
    -w /opt/work \
    -it rapidsai/citestwheel:26.04-cuda12.9.1-ubuntu22.04-py3.11 \
    bash

ci/test_wheel_cugraph-pyg.sh
