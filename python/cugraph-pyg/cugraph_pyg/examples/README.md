Examples
--------

To run the examples, first launch the latest [PyG container from NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pyg):

```bash
cd $GIT_ROOT

docker run \
  -it \
  --gpus all \
  -v $PWD:/workspace/cugraph-gnn \
  nvcr.io/nvidia/pyg:25.10-py3 \
  bash
```

```bash
cd /workspace/cugraph-gnn

torchrun \
  --nnodes 1 \
  --nproc-per-node 2 \
  python/cugraph-pyg/cugraph_pyg/examples/movielens_mnmg.py

torchrun \
  --nnodes 1 \
  --nproc-per-node 2 \
  python/cugraph-pyg/cugraph_pyg/examples/gcn_dist_mnmg.py

```

- `--nnodes` sets the number of nodes (in this case just 1 since you're training on a single node)
- `--nproc-per-node` is the number of process run per node.  This is equal to the number of GPUs per node (in this example, just 1 GPU for single-GPU training)
