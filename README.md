# CluProp

CluProp is a C++17 and Python implementation of density-aware clustering on precomputed **approximate** k-NN graphs.
It first converts a directed k-NN graph into a symmetric k-NN graph, keeping an undirected connection whenever either point appears in the other's neighbourhood.
Clustering is performed by DANE (Density-Aware Neighborhood Expansion), which grows clusters from dense regions to sparse reagions using a priority queue. 

A candidate point $x$, reached from a higher-density predecessor $p$, is processed with the priority
$d(p,x) + d_k(x)$, where $d_k(x)$ is the local k-NN distance of $x$. 
This favors short connections into locally dense regions. 
DANE further requires local neighbourhood support before propagating a cluster label from the predecessor, helping prevent expansion across weak or spurious connections due to the approximation of kNN graph.

CluProp currently assumes the approximate k-NN graph is computed externally.
The Python extension accepts directed k-NN indices and distances, builds a symmetric graph, and propagates cluster labels through that graph.

## Requirements

- Python 3.7 or newer
- A C++17 compiler
- Eigen 3
- Boost headers
- OpenMP
- pybind11 (a copy is included in this repository)

The experiment scripts under `test/` have additional optional dependencies, including NumPy, FAISS, scikit-learn, igraph, leidenalg, and NNDescent.

## Installation

Build and install the Python extension from the repository root:

```bash
python -m pip install .
```

On Linux, the build uses OpenMP and enables `-march=native`. A wheel built on one machine may not run on a different CPU family.

On macOS, install an OpenMP runtime such as Homebrew's `libomp` before building:

```bash
brew install libomp eigen boost
python -m pip install .
```

## Python usage

`knn_dane` expects one row per point. `indices[i, j]` is the ID of a neighbour of point `i`, and `distances[i, j]` is the corresponding distance. Use `int32` indices and `float32` distances.

```python
import numpy as np
import cluprop

indices = np.array([
    [1, 2], [0, 2], [1, 0],
    [4, 5], [3, 5], [4, 3],
], dtype=np.int32)

distances = np.ones((6, 2), dtype=np.float32)

model = cluprop.cluprop()
model.knn_dane(indices, distances, k=2)

labels = np.asarray(model.labels_)
print(labels)
```

## API

```python
model = cluprop.cluprop()
model.set_threads(n_threads)
model.clear()

model.knn_dane(indices, distances, k)
labels = model.labels_
```

- `k` controls the neighbourhood size used by propagation.
- `labels_` contains one cluster label per input point.

## Project layout

```text
src/cluprop.cpp            DANE propagation and graph orchestration
src/cluprop.h              CluProp class and public C++ API
src/csr_graph.*            Symmetric k-NN graph construction and CSR storage
python/python_wrapper.cpp  pybind11 bindings
setup.py                   Python extension build configuration
test/                      Benchmark and experiment scripts
```

## CMake

The current `CMakeLists.txt` builds a minimal executable whose `main` exits immediately. To build the usable Python extension, use the Python build command above.

## License

The Python package metadata declares the MIT license. Add a root `LICENSE` file before distributing the project if one is not already supplied.

## Authors

> Yingtao Zheng, Hugo Phibbs, Ninh Pham. 
> "Scalable Density-based Clustering via Density-aware Propagation on Approximate kNN Graphs."
> ICDM 2026
