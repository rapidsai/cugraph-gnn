# cugraph_pyg

## Overview
**cugraph_pyg** brings GPU-accelerated graph processing from [RAPIDS](https://rapids.ai) cuGraph to PyTorch Geometric (PyG), enabling seamless integration of cuGraph's high-performance capabilities into the PyG ecosystem. By providing native implementations of PyG's `GraphStore`, `FeatureStore`, and `Loader` interfaces, cugraph_pyg unlocks powerful GPU-accelerated graph analytics—including neighborhood sampling, centrality metrics, and community detection—directly within your PyG workflows.

Designed for scalability, cugraph_pyg supports both single-GPU and multi-node, multi-GPU configurations, making it ideal for training large-scale Graph Neural Networks (GNNs). Simply use cugraph_pyg as a drop-in replacement for standard PyG components to accelerate your graph learning pipelines with minimal code changes.

## Usage
```
edge_index = torch.tensor([
    [0, 3, 2, 8, 1, 8, 5, 0, 7, 3],
    [4, 1, 1, 2, 7, 6, 4, 7, 8, 5]
])
x = torch.tensor([
    [0.6, 0.5, 0.4],
    [0.0, 0.1, 1.1],
    [0.1, 0.2, 0.3],
    [0.5, 0.1, 0.1],
    [0.8, 0.8, 0.7],
    [1.1, 0.1, 0.1],
    [1.2, 0.2, 0.3],
    [0.2, 0.1, 0.2],
    [0.3, 0.2, 0.5],
])
y = torch.tensor([0, 1, 0, 0, 1, 0, 0, 1, 1])

graph_store = cugraph_pyg.data.GraphStore()
feature_store = cugraph_pyg.data.FeatureStore()

graph_store[('n', 'to', 'n'), 'coo', False, (9, 9)] = edge_index
feature_store['n', 'x', None] = x
feature_store['n', 'y', None] = y

loader = cugraph_pyg.data.NeighborLoader(
    (feature_store, graph_store),
    input_nodes=torch.arange(9),
    num_neighbors=[2, 2],
    batch_size=2,
)

for batch in loader:
    ...
...
```
