# GNN Diagnostics Investigation

```bash
#!/bin/bash

docker run \
  -it \
  --gpus all \
  -e LOCAL_WORLD_SIZE=1 \
  -v $PWD:/workspace \
  -v ~/.cache:/root/.cache \
  -w /workspace \
  rapidsai/ci-conda:25.12-cuda13.0.1-rockylinux8-py3.12 \
  bash

mamba install -y -c pytorch -c nvidia pytorch==2.8.0
mamba install -y -c rapidsai-nightly "cugraph-pyg=25.12.*" "pylibwholegraph=25.12.*" "cudf=25.12.*" "cuml=25.12.*"
pip install matplotlib seaborn
```

```bash
python synthetic_diagnostics_demo.py \
  --device cuda \
  --epochs 1000 \
  --hessian-sample-every 5 \
  --hessian-max-samples 200 \
  --plateau-step 500 \
  --plateau-lr-scale 0.1
```
