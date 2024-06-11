<h1 align="center"; style="font-style: italic";>
  <br>
  <img src="img/cugraph_logo_2.png" alt="cuGraph" width="500">
</h1>

<div align="center">

<a href="https://github.com/rapidsai/cugraph-gnn/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
<img alt="GitHub tag (latest by date)" src="https://img.shields.io/github/v/tag/rapidsai/cugraph-gnn">

<a href="https://github.com/rapidsai/cugraph-gnn/stargazers">
    <img src="https://img.shields.io/github/stars/rapidsai/cugraph-gnn"></a>
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/rapidsai/cugraph-gnn">

<img alt="Conda [cuGraph-DGL]" src="https://img.shields.io/conda/pn/rapidsai/cugraph-dgl" />
<img alt="Conda [cuGraph-PyG]" src="https://img.shields.io/conda/pn/rapidsai/cugraph-pyg" />
<img alt="Conda [WholeGraph]" src="https://img.shields.io/conda/pn/rapidsai/wholegraph" />

<a href="https://rapids.ai/"><img src="img/rapids_logo.png" alt="RAPIDS" width="125"></a>

</div>

<br>

[RAPIDS](https://rapids.ai) cuGraph GNN is a monorepo containing packages for GPU-accelerated graph neural networks (GNNs).
cuGraph-GNN supports the creation and manipulation of graphs followed by the execution of scalable fast graph algorithms.

<div align="center">

[Getting cuGraph](./docs/cugraph/source/installation/getting_cugraph.md) *
[Graph Algorithms](./docs/cugraph/source/graph_support/algorithms.md) *
[GNN Support](./readme_pages/gnn_support.md)

</div>

-----
## News

___NEW!___   _[nx-cugraph](./python/nx-cugraph/README.md)_, a NetworkX backend that provides GPU acceleration to NetworkX with zero code change.
```
> pip install nx-cugraph-cu11 --extra-index-url https://pypi.nvidia.com
> export NETWORKX_AUTOMATIC_BACKENDS=cugraph
```
That's it.  NetworkX now leverages cuGraph for accelerated graph algorithms.

-----

## Table of contents
- Installation
  - [Getting cuGraph Packages](./docs/cugraph/source/installation/getting_cugraph.md)
  - [Building from Source](./docs/cugraph/source/installation/source_build.md)
  - [Contributing to cuGraph](./readme_pages/CONTRIBUTING.md)
- General
  - [Latest News](./readme_pages/news.md)
  - [Current list of algorithms](./docs/cugraph/source/graph_support/algorithms.md)
  - [Blogs and Presentation](./docs/cugraph/source/tutorials/cugraph_blogs.rst)
  - [Performance](./readme_pages/performance/performance.md)
- Packages
  - [cugraph-dgl](./readme_pages/cugraph_dgl.md)
  - [cugraph-pyg](./readme_pages/cugraph_dgl.md)
- API Docs
  - Python
    - [Python Nightly](https://docs.rapids.ai/api/cugraph/nightly/)
    - [Python Stable](https://docs.rapids.ai/api/cugraph/stable/)
- References
  - [RAPIDS](https://rapids.ai/)
  - [DGL](https://dgl.ai)
  - [PyG](https://pyg.org)

<br><br>

-----

<img src="img/Stack2.png" alt="Stack" width="800">

[RAPIDS](https://rapids.ai) cuGraph-GNN is a collection of GPU-accelerated plugins that support [DGL](https://dgl.ai), [PyG](https://pyg.org), [PyTorch](https://pytorch.org), and a variety
of other graph and GNN frameworks.  cuGraph-GNN is built on top of RAPIDS [cuGraph](https://github.com/rapidai/cugraph), leveraging its low-level [pylibcugraph](https://github.com/rapidsai/cugraph/python/pylibcugraph) API
and C++ primitives for sampling and other GNN operations ([libcugraph](https://github.com/rapidai/cugraph/python/libcugraph))

cuGraph-GNN is comprised of three subprojects: [cugraph-DGL](https://github.com/rapidsai/cugraph-gnn/python/cugraph-dgl), [cugraph-PyG](https://github.com/rapidsai/cugraph-gnn/python/cugraph-pyg), and
[WholeGraph](https://github.com/rapidsai/cugraph-gnn/python/wholegraph).

* cuGraph-DGL supports the Deep Graph Library (DGL) and offers duck-typed versions of DGL's native graph objects, samplers, and loaders.
* cuGraph-PyG supports PyTorch Geometric (PyG) and implements PyG's GraphStore, FeatureStore, Loader, and Sampler interfaces.
* WholeGraph supports PyTorch and provides a distributed graph and kv store.  Both cuGraph-DGL and cuGraph-PyG can leverage WholeGraph for even greater scalability.

------
# Projects that use cuGraph

(alphabetical order)
* ArangoDB - a free and open-source native multi-model database system  - https://www.arangodb.com/
* CuPy - "NumPy/SciPy-compatible Array Library for GPU-accelerated Computing with Python" -  https://cupy.dev/
* Memgraph - In-memory Graph database - https://memgraph.com/
* NetworkX (via [nx-cugraph](./python/nx-cugraph/README.md) backend) - an extremely popular, free and open-source package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks - https://networkx.org/
* PyGraphistry - free and open-source GPU graph ETL, AI, and visualization, including native RAPIDS & cuGraph support - http://github.com/graphistry/pygraphistry
* ScanPy - a scalable toolkit for analyzing single-cell gene expression data - https://scanpy.readthedocs.io/en/stable/

(please post an issue if you have a project to add to this list)



------
<br>

## <div align="center"><img src="img/rapids_logo.png" width="265px"/></div> Open GPU Data Science <a name="rapids"></a>


The RAPIDS suite of open source software libraries aims to enable execution of end-to-end data science and analytics pipelines entirely on GPUs. It relies on NVIDIA® CUDA® primitives for low-level compute optimization but exposing that GPU parallelism and high-bandwidth memory speed through user-friendly Python interfaces.

<p align="center"><img src="img/rapids_arrow.png" width="50%"/></p>

For more project details, see [rapids.ai](https://rapids.ai/).

<br><br>
### Apache Arrow on GPU  <a name="arrow"></a>

The GPU version of [Apache Arrow](https://arrow.apache.org/) is a common API that enables efficient interchange of tabular data between processes running on the GPU. End-to-end computation on the GPU avoids unnecessary copying and converting of data off the GPU, reducing compute time and cost for high-performance analytics common in artificial intelligence workloads. As the name implies, cuDF uses the Apache Arrow columnar data format on the GPU. Currently, a subset of the features in Apache Arrow are supported.
