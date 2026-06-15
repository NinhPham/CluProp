import os

# must set before import

# os.environ["MKL_NUM_THREADS"] = "8"
# os.environ["OPENBLAS_NUM_THREADS"] = "8"
# os.environ["NUMEXPR_NUM_THREADS"] = "8"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
# os.environ["OMP_NUM_THREADS"] = "8"
# os.environ["FAISS_NUM_THREADS"] = "8"

# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["FAISS_NUM_THREADS"] = "1"

# import sDbscan
import faiss
import hdbscan
import numpy as np
import math
import igraph as ig
import leidenalg

from joblib import Parallel, delayed

from sklearn.cluster import DBSCAN, OPTICS, KMeans, SpectralClustering,cluster_optics_dbscan
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from scipy.spatial.distance import jensenshannon
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.metrics.cluster import pair_confusion_matrix

from sklearn.neighbors import NearestNeighbors
import networkx as nx
import random
from collections import Counter
from collections import defaultdict

from umap import UMAP
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt

import timeit
import gc
from concurrent.futures import ThreadPoolExecutor

def print_array(A):
    for row in A:
        print(' '.join(f"{val:.3f}" for val in row))
def mmap_bin(bin_path, num_rows, num_cols, dtype=np.float32):
    # return np.memmap(bin_path, dtype=dtype, mode='r', shape=(num_rows, num_cols)) # read-only mode
    return np.memmap(bin_path, dtype=dtype, mode='c', shape=(num_rows, num_cols)) # copy-on-write mode

#==========================================================================
def getMetric(labels, true_labels):

    M = pair_confusion_matrix(labels, true_labels)
    n = np.size(labels)

    N00 = np.float32(M[0][0]) / n
    N10 = np.float32(M[1][0]) / n
    N01 = np.float32(M[0][1]) / n
    N11 = np.float32(M[1][1]) / n
    a = N11 * N00 - N10 * N01
    b = math.sqrt(N11 + N10) * math.sqrt(N11 + N01) * math.sqrt(N00 + N10) * math.sqrt(N00 + N01)

    numLabels = len(set(labels))
    nmi_score = normalized_mutual_info_score(true_labels, labels)
    ari_score = adjusted_rand_score(true_labels, labels)
    ami_score = adjusted_mutual_info_score(true_labels, labels)
    cc_score = a / b

    return np.array([numLabels, nmi_score, ari_score, ami_score, cc_score])

def extract_edges_chunk(indices, start, end):
    edges = set()
    for i in range(start, end):
        for j in indices[i]:
            if i != j:
                edges.add(tuple(sorted((i, j))))
    return edges

def build_sym_knn_graph_parallel(indices, n_jobs=32):
    n, k_max = indices.shape

    chunk_size = (n + n_jobs - 1) // n_jobs
    results = Parallel(n_jobs=n_jobs, prefer='threads')(
        delayed(extract_edges_chunk)(indices, i, min(i + chunk_size, n))
        for i in range(0, n, chunk_size)
    )
    # Merge edge sets
    all_edges = set().union(*results)

    # Create graph
    G = nx.Graph()
    G.add_edges_from(all_edges)
    return G

#============================================================================
def faiss_kmeans(X, n_clusters=10, n_threads = 8, n_iter=20, gpu=False):
    """
    X: numpy array [n_samples, n_features]
    n_clusters: number of clusters
    n_iter: number of iterations
    gpu: whether to use GPU (True/False)
    """
    X = X.astype(np.float32)
    d = X.shape[1]  # dimensionality

    faiss.omp_set_num_threads(n_threads) # This is also default
    kmeans = faiss.Kmeans(
        d=d,
        k=n_clusters,
        niter=n_iter,
        nredo=1,
        verbose=True,
        min_points_per_centroid=50,  # avoid dropping centroids
        max_points_per_centroid=1000000  # disable sampling cap
    )
    kmeans.train(X.astype(np.float32))

    # labeling by 1NN
    distances, labels = kmeans.index.search(X, 1)
    labels = labels.flatten()

    return labels

def nystrom_kernel_kmeans(X, n_clusters=10, m=1000, gamma=0.5, n_iter=10):
    """
    X: data matrix [n x d]
    n_clusters: number of clusters
    m: number of landmark points
    gamma: RBF kernel parameter
    n_iter: number of k-means iterations
    """
    n = X.shape[0]

    # 1. Landmark sampling
    X_landmarks = shuffle(X, random_state=42)[:m]

    # 2. Compute kernel blocks
    W_mm = rbf_kernel(X_landmarks, X_landmarks, gamma=gamma)
    W_nm = rbf_kernel(X, X_landmarks, gamma=gamma)

    # 3. Nyström approximation of feature map Z ≈ W_nm @ W_mm^{-1/2}
    eigvals, eigvecs = np.linalg.eigh(W_mm)
    idx = eigvals > 1e-10
    W_mm_inv_sqrt = eigvecs[:, idx] @ np.diag(1.0 / np.sqrt(eigvals[idx])) @ eigvecs[:, idx].T
    Z = W_nm @ W_mm_inv_sqrt  # [n x m]

    # 4. Run k-means on Z (approximate kernel feature space)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=n_iter)
    labels = kmeans.fit_predict(Z)

    return labels, Z

def nystrom_spectral(X, k=10, m=1000, gamma=1.0, n_iter=20):

    X_landmarks = shuffle(X, random_state=42)[:m]
    W_mm = rbf_kernel(X_landmarks, X_landmarks, gamma=gamma)
    W_nm = rbf_kernel(X, X_landmarks, gamma=gamma)

    # Eigen-decomposition on W_mm
    eigvals, eigvecs = np.linalg.eigh(W_mm)
    top_idx = np.argsort(eigvals)[-k:]
    U = eigvecs[:, top_idx]

    # Out-of-sample extension
    Z = W_nm @ U
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)

    # Cluster with k-means
    labels = KMeans(n_clusters=k, max_iter=n_iter).fit_predict(Z)
    return labels

def build_knn_graph(
        X,
        k=10,
        metric="euclidean",
        mutual=False,
        include_self=False,
        n_jobs=None,
        algorithm="auto"
):
    """
    Build a sparse kNN adjacency (unweighted, 1s) using sklearn NearestNeighbors.

    Returns
    -------
    A : scipy.sparse.csr_matrix, shape (n, n)
        Unweighted adjacency (0/1). If mutual=False -> symmetric-kNN (A = A ∪ A^T).
        If mutual=True -> mutual-kNN (A = A ∩ A^T).
    dists : scipy.sparse.csr_matrix
        Sparse matrix with neighbor distances (same sparsity as A before symmetrization).
        Will be symmetrized consistently with A.
    """
    n = X.shape[0]
    # Note: n_neighbors = k + (self if included)
    n_neighbors = k + (1 if include_self else 0)

    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric,
        algorithm=algorithm,
        n_jobs=n_jobs
    ).fit(X)
    distances, indices = nbrs.kneighbors(X, return_distance=True)

    # Optionally drop self edges (distance 0)
    if include_self:
        distances = distances[:, 1:]
        indices = indices[:, 1:]

    # Build row-wise sparse from neighbors
    row = np.repeat(np.arange(n), k)
    col = indices.ravel()
    data = np.ones_like(col, dtype=np.float64)

    A = sparse.csr_matrix((data, (row, col)), shape=(n, n))
    D = sparse.csr_matrix((distances.ravel(), (row, col)), shape=(n, n))

    if mutual:
        # Mutual-kNN: A_mutual = A ∩ A^T
        A = A.minimum(A.T)
        # Distances for mutual edges: take max of the two directed distances (safer) or min.
        D = D.maximum(D.T).multiply(A)  # keep only mutual edges
    else:
        # Symmetric-kNN: A_sym = A ∪ A^T
        A = A.maximum(A.T)
        # Distances: choose symmetric distance (min is common)
        D = D.minimum(D.T).multiply(A)

    A.eliminate_zeros()
    D.eliminate_zeros()
    return A, D
def gaussian_weight(D, sigma="auto"):
    """
    Convert a sparse distance matrix to weighted adjacency with Gaussian kernel:
        w_ij = exp( -d_ij^2 / (2 * sigma^2) )

    sigma:
        - float: global sigma
        - "auto": median of nonzero distances / sqrt(2) (robust default)
        - ("median-k", c): use median of each node's k-NN distances averaged, divided by c
    """
    if isinstance(sigma, tuple) and sigma[0] == "median-k":
        # Expect D has exactly k neighbors per node pre-symmetrization; robust anyway
        # Compute per-row median of nonzeros, then average
        medians = []
        for i in range(D.shape[0]):
            start, end = D.indptr[i], D.indptr[i + 1]
            vals = D.data[start:end]
            if vals.size:
                medians.append(np.median(vals))
        gsigma = (np.median(medians) / float(sigma[1])) if medians else 1.0
    elif sigma == "auto":
        vals = D.data
        vals = vals[vals > 0]
        gsigma = (np.median(vals) / np.sqrt(2.0)) if vals.size else 1.0
    else:
        gsigma = float(sigma)

    if gsigma <= 0:
        gsigma = 1.0

    W = D.copy()
    W.data = np.exp(-(W.data ** 2) / (2.0 * gsigma * gsigma))
    W.eliminate_zeros()
    return W
def normalized_laplacian(W, which="sym"):
    """
    Build normalized Laplacian:
      L_sym = I - D^{-1/2} W D^{-1/2}
      L_rw  = I - D^{-1} W

    Returns sparse L.
    """
    deg = np.asarray(W.sum(axis=1)).ravel()
    if which == "sym":
        with np.errstate(divide="ignore"):
            d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
        D_inv_sqrt = sparse.diags(d_inv_sqrt)
        L = sparse.eye(W.shape[0], format="csr") - D_inv_sqrt @ W @ D_inv_sqrt
    elif which == "rw":
        with np.errstate(divide="ignore"):
            d_inv = 1.0 / np.maximum(deg, 1e-12)
        D_inv = sparse.diags(d_inv)
        L = sparse.eye(W.shape[0], format="csr") - D_inv @ W
    else:
        raise ValueError("which must be 'sym' or 'rw'")
    L.eliminate_zeros()
    return L
def bottom_k_eigenvectors(L, k, tol=1e-4, maxiter=None, random_state=0):
    """
    Compute the k smallest eigenpairs of symmetric PSD L using eigsh.
    Skips the trivial 0-eigenvector (constant) by requesting k+1 and dropping it when needed.
    """
    # Request k+1 to avoid the constant vector; we'll drop it if present
    m = k + 1
    m = min(m, L.shape[0] - 1)  # safety
    vals, vecs = eigsh(L, k=m, which="SM", tol=tol, maxiter=maxiter, v0=np.random.RandomState(random_state).randn(L.shape[0]))
    # Sort
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    # Drop the smallest (≈0) eigenpair
    vals, vecs = vals[1:k+1], vecs[:, 1:k+1]
    return vals, vecs
def spectral_clustering(
        X,
        n_clusters,
        k=10,
        metric="euclidean",
        mutual=False,
        sigma="auto",
        laplacian="sym",
        random_state=0,
        n_init=10
):
    """
    Full pipeline: kNN graph -> Gaussian weights -> normalized Laplacian -> bottom-k eigenvectors -> k-means.
    Returns labels (for all points). If the graph is disconnected, clusters are found on the largest CC and the rest labeled -1.
    """
    # 1) kNN graph (unweighted) + distances
    A, D = build_knn_graph(X, k=k, metric=metric, mutual=mutual)

    # 2) Weighting
    W = gaussian_weight(D, sigma=sigma)
    # Ensure symmetry
    W = W.maximum(W.T)
    W.eliminate_zeros()

    # 3) Work on the largest connected component (spectral clustering assumes connectivity)
    n = W.shape[0]
    n_cc, labels_cc = connected_components(W, directed=False)
    if n_cc > 1:
        # pick the largest CC
        sizes = np.bincount(labels_cc)
        cc_id = np.argmax(sizes)
        mask = (labels_cc == cc_id)
    else:
        mask = np.ones(n, dtype=bool)

    W_cc = W[mask][:, mask]

    # 4) Laplacian and eigen-embedding
    L = normalized_laplacian(W_cc, which=laplacian)
    _, U = bottom_k_eigenvectors(L, k=n_clusters, random_state=random_state)

    # 5) Row-normalize embedding (Ng–Jordan–Weiss) for L_sym
    if laplacian == "sym":
        U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)

    # 6) k-means in the embedding
    km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state)
    labels_sub = km.fit_predict(U)

    # 7) Map back to full set
    labels = -np.ones(n, dtype=int)
    labels[mask] = labels_sub
    return labels

#============================================================================
# igraph
def run_louvain(graph):
    """Run Louvain algorithm and return cluster labels."""
    partition = graph.community_multilevel()
    labels = [0] * graph.vcount()
    for cid, cluster in enumerate(partition):
        for node in cluster:
            labels[node] = cid
    return labels

def run_leiden(graph, resolution=1.0):
    """Run Leiden algorithm and return cluster labels."""
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=None
    )

    # Run Leiden community detection
    # partition = graph.community_leiden(weights='weight', resolution_parameter=resolution) # bug so do not know how to fix

    labels = [0] * graph.vcount()
    for cid, cluster in enumerate(partition):
        for node in cluster:
            labels[node] = cid
    return labels

def run_LPA(G):
    """
    Run label propagation on an iGraph object and return cluster labels.
    """

    # Step 3: Run Label Propagation
    communities = G.community_label_propagation()
    labels = [0] * G.vcount()
    for cluster_id, cluster in enumerate(communities):
        for node in cluster:
            labels[node] = cluster_id

    return labels

#============================================================================
# networkx
def nx_LPA(G, max_iter=100):

    labels = {node: node for node in G.nodes()}
    for _ in range(max_iter):
        nodes = list(G.nodes())
        random.shuffle(nodes)

        updated = False
        for node in nodes:
            neighbor_labels = [labels[nbr] for nbr in G.neighbors(node)]
            if not neighbor_labels:
                continue
            most_common = Counter(neighbor_labels).most_common(1)[0][0]
            if labels[node] != most_common:
                labels[node] = most_common
                updated = True

        if not updated:
            break

    clusters = {}
    for node, label in labels.items():
        clusters.setdefault(label, []).append(node)

    nodes = list(G.nodes())
    point_to_cluster = {}

    for cluster_id, nodes_in_cluster in enumerate(clusters.values()):
        for node in nodes_in_cluster:
            point_to_cluster[node] = cluster_id

    # Preserve original node order
    labels = [point_to_cluster[node] for node in nodes]



    return labels

#============================================================================
def getAcc_kNNG(exact, approx):
    n, k = np.shape(exact)
    result = 0
    for i in range(n):
        result += len(np.intersect1d(exact[i], approx[i])) / k
    return result / n

def faiss_approx_kNN_IVF(X, k=10, n_list = 100, n_probe = 10, n_threads=8):
    """
    Run label propagation clustering using Faiss + iGraph.

    Parameters:
    - X: np.ndarray of shape (n, d)
    - k: number of nearest neighbors (default: 10)
    - metric: 'squared_l2' or 'dot_product'

    Returns:
    - labels: list of cluster labels for each point
    """

    X = X.astype(np.float32)
    n, d = X.shape

    # 1. Create FAISS index
    faiss.omp_set_num_threads(n_threads) # This is also default
    nlist = n_list  # the number of clusters
    print("nlist = ", nlist)

    t1 = timeit.default_timer()
    quantizer = faiss.IndexFlatL2(d)  # the other index
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    index.nprobe = n_probe
    print("nprobe = ", n_probe)
    # 8 specifies that each sub-vector is encoded as 8 bits
    index.train(X)
    index.add(X)

    t2 = timeit.default_timer()
    print('Construction time of Faiss IVF: {}'.format(t2 - t1))

    batch_size = 10000
    distances = np.empty((n, k), dtype='float32')
    indices = np.empty((n, k), dtype='int64')  # Faiss returns int64 indices

    for i in range(0, X.shape[0], batch_size):
        end = min(i + batch_size, n)
        D, I = index.search(X[i:i+batch_size], k)
        distances[i:end] = D
        indices[i:end] = I

    # distances, indices = index.search(X, k)

    t2 = timeit.default_timer()
    print('Build and query time of Faiss IVF: {}'.format(t2 - t1))

    return indices, distances

def faiss_approx_kNN_IVFPQ(X, k=10, n_list = 100, n_subquantizer = 8, n_probe = 10, n_threads=8):
    """
    Run label propagation clustering using Faiss + iGraph.

    Parameters:
    - X: np.ndarray of shape (n, d)
    - k: number of nearest neighbors (default: 10)
    - metric: 'squared_l2' or 'dot_product'

    Returns:
    - labels: list of cluster labels for each point
    """

    X = X.astype(np.float32)
    n, d = X.shape

    # 1. Create FAISS index
    faiss.omp_set_num_threads(n_threads) # This is also default
    nlist = n_list  # the number of clusters
    print("nlist = ", nlist) # number of coarse centroids (IVF)

    m = n_subquantizer  # number of PQ subquantizers
    nbits = 8  # bits per subquantizer


    t1 = timeit.default_timer()
    quantizer = faiss.IndexFlatL2(d)  # the other index
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)

    index.nprobe = n_probe
    print("nprobe = ", n_probe)

    # 8 specifies that each sub-vector is encoded as 8 bits
    index.train(X)
    index.add(X)

    t2 = timeit.default_timer()
    print('Construction time of Faiss IVFPQ: {}'.format(t2 - t1))

    batch_size = 10000
    distances = np.empty((n, k), dtype='float32')
    indices = np.empty((n, k), dtype='int32')  # Faiss returns int64 indices

    for i in range(0, X.shape[0], batch_size):
        end = min(i + batch_size, n)
        D, I = index.search(X[i:i+batch_size], k)
        distances[i:end] = D
        indices[i:end] = I

    # distances, indices = index.search(X, k)

    t2 = timeit.default_timer()
    print('Build and query time of Faiss IVFPQ: {}'.format(t2 - t1))

    return indices, distances

def faiss_kNN(X, k=10, n_threads=8):
    """
    Run label propagation clustering using Faiss + iGraph.

    Parameters:
    - X: np.ndarray of shape (n, d)
    - k: number of nearest neighbors (default: 10)
    - metric: 'squared_l2' or 'dot_product'

    Returns:
    - labels: list of cluster labels for each point
    """

    X = X.astype(np.float32)
    n, d = X.shape

    # 1. Create FAISS index
    faiss.omp_set_num_threads(n_threads) # This is also default

    t1 = timeit.default_timer()
    index = faiss.IndexFlatL2(d)  # L2 distance (Euclidean)
    index.add(X)
    distances, indices = index.search(X, k)

    indices = indices.astype(np.int32)
    distances = distances.astype(np.float32)  # optional, if you need float32

    t2 = timeit.default_timer()

    print('Exact kNN time : {}'.format(t2 - t1))

    return indices, distances

def scann_approx_kNN(X, k=10, n_list = 100, n_probe = 10):
    """
    Run label propagation clustering using Faiss + iGraph.

    Parameters:
    - X: np.ndarray of shape (n, d)
    - k: number of nearest neighbors (default: 10)
    - metric: 'squared_l2' or 'dot_product'

    Returns:
    - labels: list of cluster labels for each point
    """

    X = X.astype(np.float32)
    n, d = X.shape

    print("nlist = ", n_list)
    print("nprobe = ", n_probe)

    t1 = timeit.default_timer()
    searcher = scann.scann_ops_pybind.builder(X, k, "dot_product").tree(
        num_leaves=n_list, num_leaves_to_search=n_probe, training_sample_size=n).score_ah(
        2, anisotropic_quantization_threshold=0.2).reorder(100).build()
    t2 = timeit.default_timer()
    print('Construction time of Scann: {}'.format(t2 - t1))

    indices, distances = searcher.search_batched(X)
    indices = indices.astype(np.int32)
    distances = distances.astype(np.float32)  # optional, if you need float32

    t2 = timeit.default_timer()
    print('Build and query time of Scann: {}'.format(t2 - t1))

    return indices, distances

#============================================================================

def igraph_form_unweighted_sym_KNN_graph(indices, verbose=False):

    t1 = timeit.default_timer()
    n = len(indices)

    edge_set = set()

    for i in range(n):
        for j in indices[i]:
            if i == j or j < 0:
                continue
            edge = tuple(sorted((i, j)))  # undirected
            edge_set.add(edge)

    # Step 2: Build igraph
    G = ig.Graph(n=n, edges=list(edge_set), directed=False)

    t2 = timeit.default_timer()
    if (verbose):
        print('Form unweighted graph time: {}'.format(t2 - t1))

    return G

def fast_unweighted_sym_knng_igraph(indices, verbose=False):
    """
    Construct a weighted symmetric graph for igraph from Faiss kNN output.

    Parameters:
    - indices: np.ndarray of shape (n, k)
    - distances: np.ndarray of shape (n, k)
    - use_exp_weight: if True, use exp(-dist); else use 1 / (dist + eps)

    Returns:
    - igraph.Graph with 'weight' edge attribute
    """

    t1 = timeit.default_timer()
    n, k = indices.shape

    # Repeat source indices
    src = np.repeat(np.arange(n), k)
    dst = indices.reshape(-1)

    # Remove self-loops and invalid (-1) neighbors
    mask = (dst >= 0) & (src != dst)
    src, dst = src[mask], dst[mask]

    # Canonical edge (min, max) for undirected deduplication
    a = np.minimum(src, dst)
    b = np.maximum(src, dst)
    edges = np.vstack((a, b)).T

    # Remove duplicate edges (symmetric kNN)
    # Use np.unique for speed
    edge_keys = a * n + b  # unique encoding
    unique_keys, unique_idx = np.unique(edge_keys, return_index=True)

    edges = edges[unique_idx]

    # Build igraph
    G = ig.Graph(edges=edges.tolist(), directed=False)

    t2 = timeit.default_timer()
    if (verbose):
        print('Form unweighted graph time: {}'.format(t2 - t1))

    return G

def igraph_form_weighted_sym_KNN_graph(indices, distances, use_exp_weight=False, sym_op='mean', verbose=False):

    """
    Build a symmetric kNN graph from FAISS/ScaNN output for igraph.

    Parameters:
    - indices: list or array of shape (n, k), neighbor indices
    - distances: same shape as indices, corresponding distances
    - weighted: bool, whether to compute edge weights
    - sym_op: 'max', 'min', or 'mean' to merge duplicate edges

    Returns:
    - igraph.Graph with or without weights
    """

    t1 = timeit.default_timer()
    eps=1e-5
    n = len(indices)

    edge_weights = defaultdict(list)

    for i in range(n):
        for j, d in zip(indices[i], distances[i]):
            if i == j or j < 0:
                continue
            if d == 0:
                d = eps
            a, b = sorted((i, j))
            edge_weights[(a, b)].append(d)

    edges = []
    weights = []

    for (i, j), dists in edge_weights.items():
        edges.append((i, j))

        dist = (
            max(dists) if sym_op == "max"
            else min(dists) if sym_op == "min"
            else sum(dists) / len(dists)
        )
        # weight = np.exp(-dist)  # or 1 / (dist + ε)
        weight = np.exp(-dist) if use_exp_weight else 1.0 / (dist + eps)
        weights.append(weight)

    G = ig.Graph(edges=edges, directed=False)
    G.es["weight"] = weights

    t2 = timeit.default_timer()
    if (verbose):
        print('Form weighted graph time: {}'.format(t2 - t1))

    return G

def igraph_form_weighted_sym_KNN_graph_fast(indices, distances, use_exp_weight=False, sym_op='mean', verbose=False):

    """
    Build a symmetric kNN graph from FAISS/ScaNN output for igraph.

    Parameters:
    - indices: list or array of shape (n, k), neighbor indices
    - distances: same shape as indices, corresponding distances
    - weighted: bool, whether to compute edge weights
    - sym_op: 'max', 'min', or 'mean' to merge duplicate edges

    Returns:
    - igraph.Graph with or without weights
    """

    t1 = timeit.default_timer()
    eps=1e-5
    n = len(indices)

    edges = []
    weights = []

    edge_set = set()

    for i in range(n):
        for j, d in zip(indices[i], distances[i]):

            if i == j or j < 0:
                continue

            a, b = sorted((i, j))

            if (a, b) not in edge_set:
                edge_set.add((a, b))

                edges.append((a, b))

                weight = np.exp(-d) if use_exp_weight else 1.0 / (d + eps)
                weights.append(weight)

    G = ig.Graph(edges=edges, directed=False)
    G.es["weight"] = weights

    t2 = timeit.default_timer()
    if (verbose):
        print('Form weighted graph time: {}'.format(t2 - t1))

    return G

def fast_weighted_sym_knng_igraph(indices, distances, use_exp_weight=False, eps=1e-5, verbose=False):
    """
    Construct a weighted symmetric graph for igraph from Faiss kNN output.

    Parameters:
    - indices: np.ndarray of shape (n, k)
    - distances: np.ndarray of shape (n, k)
    - use_exp_weight: if True, use exp(-dist); else use 1 / (dist + eps)

    Returns:
    - igraph.Graph with 'weight' edge attribute
    """

    t1 = timeit.default_timer()
    n, k = indices.shape

    # Repeat source indices
    src = np.repeat(np.arange(n), k)
    dst = indices.reshape(-1)
    dist = distances.reshape(-1)

    # Remove self-loops and invalid (-1) neighbors
    mask = (dst >= 0) & (src != dst)
    src, dst, dist = src[mask], dst[mask], dist[mask]

    # Canonical edge (min, max) for undirected deduplication
    a = np.minimum(src, dst)
    b = np.maximum(src, dst)
    edges = np.vstack((a, b)).T

    # Compute weights
    weights = np.exp(-dist) if use_exp_weight else 1.0 / (dist + eps)

    # Remove duplicate edges (symmetric kNN)
    # Use np.unique for speed
    edge_keys = a * n + b  # unique encoding
    unique_keys, unique_idx = np.unique(edge_keys, return_index=True)

    edges = edges[unique_idx]
    weights = weights[unique_idx]

    # Build igraph
    G = ig.Graph(edges=edges.tolist(), directed=False)
    G.es["weight"] = weights.tolist()

    t2 = timeit.default_timer()
    if (verbose):
        print('Form weighted graph time: {}'.format(t2 - t1))

    return G

def fast_local_scaled_sym_knng_igraph(
    indices,
    distances,
    mode="local_exp",
    eps=1e-12,
    verbose=False,
):
    """
    Construct a locally scaled weighted symmetric kNN graph for igraph
    from Faiss / kNN output.

    Parameters
    ----------
    indices : np.ndarray, shape (n, k)
        indices[i, t] is the t-th nearest neighbor of point i.

    distances : np.ndarray, shape (n, k)
        distances[i, t] is the distance from point i to indices[i, t].

        Important:
        - If using Faiss IndexFlatL2, these are usually squared L2 distances.
        - If using standard sklearn kNN, these are usually unsquared distances.
        Choose mode accordingly.

    mode : str
        Weighting scheme.

        "local_exp":
            w_ij = exp( - d_ij^2 / (sigma_i sigma_j) )
            Use this when distances are ordinary distances.

        "local_exp_faiss_l2":
            w_ij = exp( - D_ij / sqrt(sigma_i_sq sigma_j_sq) )
            Use this when distances are squared L2 distances from Faiss.

        "local_inverse":
            w_ij = 1 / ( d_ij / sqrt(sigma_i sigma_j) + eps )
            Use this if you want inverse locally normalized distance.

        "inverse":
            w_ij = 1 / (d_ij + eps)
            Original baseline.

        "exp":
            w_ij = exp(-d_ij)
            Original exponential baseline.

    eps : float
        Numerical stability constant.

    verbose : bool
        Whether to print runtime.

    Returns
    -------
    G : igraph.Graph
        Undirected weighted graph with edge attribute "weight".

    sigma : np.ndarray, shape (n,)
        Local scale of each point.
    """

    t1 = timeit.default_timer()

    indices = np.asarray(indices)
    distances = np.asarray(distances)

    n, k = indices.shape

    # Local scale sigma_i = distance to k-th nearest neighbor.
    # If distances are squared L2, sigma is also squared scale.
    sigma = distances[:, -1].astype(np.float64)
    sigma = np.maximum(sigma, eps)

    # Source and destination arrays.
    src = np.repeat(np.arange(n), k)
    dst = indices.reshape(-1)
    dist = distances.reshape(-1).astype(np.float64)

    # Remove self-loops and invalid neighbors.
    mask = (dst >= 0) & (src != dst)
    src = src[mask]
    dst = dst[mask]
    dist = dist[mask]

    # Canonical undirected edge representation.
    a = np.minimum(src, dst)
    b = np.maximum(src, dst)

    # Local scales for edge endpoints.
    sigma_src = sigma[src]
    sigma_dst = sigma[dst]

    if mode == "local_exp":
        # Use when dist is ordinary distance.
        denom = sigma_src * sigma_dst + eps
        weights = np.exp(-(dist ** 2) / denom)

    elif mode == "local_exp_faiss_l2":
        # Use when dist is squared L2 distance from Faiss.
        # sigma_src and sigma_dst are also squared kNN distances.
        denom = np.sqrt(sigma_src * sigma_dst) + eps
        weights = np.exp(-dist / denom)

    elif mode == "local_inverse":
        # Locally normalized inverse distance.
        denom = np.sqrt(sigma_src * sigma_dst) + eps
        normalized_dist = dist / denom
        weights = 1.0 / (normalized_dist + eps)

    elif mode == "inverse":
        weights = 1.0 / (dist + eps)

    elif mode == "exp":
        weights = np.exp(-dist)

    else:
        raise ValueError(
            "mode must be one of: "
            "'local_exp', 'local_exp_faiss_l2', "
            "'local_inverse', 'inverse', 'exp'."
        )

    # Deduplicate symmetric kNN edges.
    # If edge appears twice, keep the larger weight.
    edge_keys = a.astype(np.int64) * np.int64(n) + b.astype(np.int64)

    order = np.argsort(edge_keys)
    edge_keys = edge_keys[order]
    a = a[order]
    b = b[order]
    weights = weights[order]

    unique_mask = np.empty(len(edge_keys), dtype=bool)
    unique_mask[0] = True
    unique_mask[1:] = edge_keys[1:] != edge_keys[:-1]

    # Start indices of each unique edge group.
    group_starts = np.flatnonzero(unique_mask)

    # Keep max weight among duplicate directed occurrences.
    max_weights = np.maximum.reduceat(weights, group_starts)

    unique_a = a[group_starts]
    unique_b = b[group_starts]

    edges = np.column_stack((unique_a, unique_b))

    G = ig.Graph(n=n, edges=edges.tolist(), directed=False)
    G.es["weight"] = max_weights.tolist()

    t2 = timeit.default_timer()

    if verbose:
        print(f"Form locally scaled weighted graph time: {t2 - t1:.3f}s")
        print(f"n = {n}, k = {k}, edges = {G.ecount()}")

    return G #, sigma # (not return sigma)

def fast_weighted_mutual_knng_igraph(indices, distances, use_exp_weight=False, eps=1e-5, verbose=False):
    """
    Construct a weighted mutual kNN graph for igraph from approximate kNN output.

    An undirected edge (i, j) is kept only if:
        j is in kNN(i) and i is in kNN(j)

    Parameters
    ----------
    indices : np.ndarray of shape (n, k)
        Neighbor indices for each point.
    distances : np.ndarray of shape (n, k)
        Corresponding neighbor distances.
    use_exp_weight : bool
        If True, use exp(-dist); otherwise use 1 / (dist + eps).
    eps : float
        Small constant to avoid division by zero.
    verbose : bool
        If True, print timing information.

    Returns
    -------
    igraph.Graph
        Undirected weighted mutual kNN graph with edge attribute 'weight'.
    """

    t1 = timeit.default_timer()
    n, k = indices.shape

    # Flatten kNN lists
    src = np.repeat(np.arange(n), k)
    dst = indices.reshape(-1)
    dist = distances.reshape(-1)

    # Remove invalid neighbors and self-loops
    mask = (dst >= 0) & (src != dst)
    src = src[mask]
    dst = dst[mask]
    dist = dist[mask]

    pairs = np.empty(src.shape[0], dtype=[("u", np.int64), ("v", np.int64)])
    pairs["u"] = src
    pairs["v"] = dst

    rev_pairs = np.empty(src.shape[0], dtype=[("u", np.int64), ("v", np.int64)])
    rev_pairs["u"] = dst
    rev_pairs["v"] = src

    mutual_mask = np.isin(pairs, rev_pairs)

    src = src[mutual_mask]
    dst = dst[mutual_mask]
    dist = dist[mutual_mask]

    # Canonical undirected representation
    a = np.minimum(src, dst)
    b = np.maximum(src, dst)

    # Compute weights
    weights = np.exp(-dist) if use_exp_weight else 1.0 / (dist + eps)

    # Deduplicate undirected edges
    edge_keys = a * n + b
    unique_keys, inverse = np.unique(edge_keys, return_inverse=True)

    # If both directions exist, choose how to combine weights.
    # Here we take the maximum weight, which corresponds to the smaller distance.
    agg_weights = np.zeros(len(unique_keys), dtype=np.float64)
    np.maximum.at(agg_weights, inverse, weights)

    edges = np.column_stack((unique_keys // n, unique_keys % n))

    # Build igraph
    G = ig.Graph(n=n, edges=edges.tolist(), directed=False)
    G.es["weight"] = agg_weights.tolist()

    t2 = timeit.default_timer()
    if verbose:
        print("Form mutual weighted graph time: {}".format(t2 - t1))

    return G

# New version on 25 Mar 2026
def fast_weighted_sym_knng_igraph_large(indices, distances, use_exp_weight=False, eps=1e-5, verbose=False):
    """
    Construct a weighted symmetric kNN graph (OR rule) for igraph from ANN output.

    Keep undirected edge (i, j) if:
        j in kNN(i) OR i in kNN(j)

    Parameters
    ----------
    indices : np.ndarray of shape (n, k)
        Neighbor indices.
    distances : np.ndarray of shape (n, k)
        Corresponding neighbor distances.
    use_exp_weight : bool
        If True, weight = exp(-dist); else weight = 1 / (dist + eps).
    eps : float
        Small constant for inverse-distance weights.
    verbose : bool
        If True, print timing info.

    Returns
    -------
    G : igraph.Graph
        Undirected weighted graph with edge attribute 'weight'.
    """
    t1 = timeit.default_timer()

    n, k = indices.shape

    # Flatten directed kNN edges
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = indices.reshape(-1).astype(np.int64, copy=False)
    dist = distances.reshape(-1)

    # Remove invalid neighbors and self-loops
    mask = (dst >= 0) & (src != dst)
    src = src[mask]
    dst = dst[mask]
    dist = dist[mask]

    # Canonical undirected form: (min(i,j), max(i,j))
    a = np.minimum(src, dst)
    b = np.maximum(src, dst)

    # Encode each undirected edge as a single integer key
    # Assumes n*n fits in int64, which is fine for typical usage.
    edge_keys = a * n + b

    # Compute weights from directed distances
    weights = np.exp(-dist) if use_exp_weight else 1.0 / (dist + eps)

    # Sort by edge key so duplicates become consecutive
    order = np.argsort(edge_keys, kind="mergesort")
    edge_keys = edge_keys[order]
    weights = weights[order]

    # Find start of each unique edge block
    unique_start = np.empty(edge_keys.shape[0], dtype=bool)
    unique_start[0] = True
    unique_start[1:] = edge_keys[1:] != edge_keys[:-1]

    unique_idx = np.flatnonzero(unique_start)
    unique_keys = edge_keys[unique_idx]

    # Aggregate weights for duplicate directed edges
    # OR-symmetric graph means both i->j and j->i collapse to one undirected edge.
    # We take the maximum weight (equiv. smallest distance).
    max_weights = np.maximum.reduceat(weights, unique_idx)

    # Decode edges back to (u, v)
    edges_u = unique_keys // n
    edges_v = unique_keys % n
    edges = np.column_stack((edges_u, edges_v))

    # Build graph
    G = ig.Graph(n=n, edges=edges.tolist(), directed=False)
    G.es["weight"] = max_weights.tolist()

    t2 = timeit.default_timer()
    if verbose:
        print(f"Form symmetric weighted graph time: {t2 - t1:.4f}s")

    return G

def fast_weighted_sym_knng_igraph_large_mem(indices, distances, use_exp_weight=False, eps=1e-5, verbose=False):
    import numpy as np
    import igraph as ig
    import timeit

    t1 = timeit.default_timer()
    n, k = indices.shape

    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = indices.ravel().astype(np.int64, copy=False)
    dist = distances.ravel()

    mask = (dst >= 0) & (src != dst)
    src = src[mask]
    dst = dst[mask]
    dist = dist[mask]

    src2 = np.minimum(src, dst)
    dst2 = np.maximum(src, dst)
    del src, dst

    keys = src2 * n + dst2
    del src2, dst2

    weights = np.exp(-dist) if use_exp_weight else 1.0 / (dist + eps)
    del dist

    order = np.argsort(keys, kind="mergesort")
    keys = keys[order]
    weights = weights[order]

    boundary = np.empty(len(keys), dtype=bool)
    boundary[0] = True
    boundary[1:] = keys[1:] != keys[:-1]
    starts = np.flatnonzero(boundary)

    unique_keys = keys[starts]
    agg_weights = np.maximum.reduceat(weights, starts)

    u = unique_keys // n
    v = unique_keys % n
    edges = np.column_stack((u, v))

    G = ig.Graph(n=n, edges=edges.tolist(), directed=False)
    G.es["weight"] = agg_weights.tolist()

    t2 = timeit.default_timer()
    if verbose:
        print(f"Form symmetric weighted graph time: {t2 - t1:.4f}s")

    return G

#============================================================================
def nx_form_unweighted_KNN_graph_indices(indices):

    n = len(indices)

    # 3. Build undirected graph
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in indices[i]:  # skip self-match
            if i == j or j < 0:
                continue
            G.add_edge(i, j)

    return G

def nx_form_unweighted_sym_KNN_graph_indices(indices):

    n = len(indices)

    edge_set = set()
    for i in range(n):
        for j in indices[i]:
            if i != j:
                edge = tuple(sorted((i, j)))
                edge_set.add(edge)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edge_set)

    return G

def nx_form_unweighted_mutual_KNN_graph_indices(indices):

    n = len(indices)

    # Build neighbor sets
    neighbors = [set(row[row != i]) for i, row in enumerate(indices)]

    # 3. Build undirected graph
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in neighbors[i]:
            if i in neighbors[j] and i < j:
                G.add_edge(i, j)

    return G

#============================================================================

def nx_form_unweighted_KNN_graph(X, k=10, n_threads=8):

    X = X.astype(np.float32)
    n, d = X.shape

    indices, distances = faiss_kNN(X, k + 1, n_threads=n_threads)

    # 3. Build undirected graph
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in indices[i]:
            if i == j:
                continue
            G.add_edge(i, j)

    # print('number of nodes: ', G.number_of_nodes())
    return G

def nx_form_unweighted_sym_KNN_graph(X, k=10, n_threads=8):

    X = X.astype(np.float32)
    n, d = X.shape

    indices, distances = faiss_kNN(X, k + 1, n_threads=n_threads)

    edge_set = set()

    for i in range(n):
        for j in indices[i]:
            if i == j:
                continue
            # Add edge (min, max) to ensure symmetry
            edge = tuple(sorted((i, j)))
            edge_set.add(edge)


    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edge_set)

    return G

def nx_form_unweighted_mutual_KNN_graph(X, k=10, n_threads=8):

    X = X.astype(np.float32)
    n, d = X.shape

    indices, distances = faiss_kNN(X, k + 1, n_threads=n_threads)

    # Build neighbor sets
    neighbors = [set(row[row != i]) for i, row in enumerate(indices)]

    # 3. Build undirected graph
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in neighbors[i]:
            if i in neighbors[j] and i < j:
                G.add_edge(i, j)

    return G

#===========================================================================================================

def nx_form_approx_unweighted_sym_KNN_graph_Faiss(X, k=10, n_list = 100, n_probe = 10, n_threads=8):
    """
    Build an undirected k-NN graph using FAISS for neighbor search.
    X must be a float32 NumPy array.
    """

    X = X.astype(np.float32)
    n, d = X.shape

    indices, distances = faiss_approx_kNN_IVF(X, k + 1, n_list=n_list, n_probe=n_probe, n_threads=n_threads)

    # 3. Build undirected graph
    return build_sym_knn_graph_parallel(indices, n_jobs=n_threads)

def nx_form_approx_unweighted_sym_KNN_graph_Scann(X, k=10, n_list = 100, n_probe = 10, n_threads=8):

    X = X.astype(np.float32)
    n, d = X.shape

    indices, distances = scann_approx_kNN(X, k + 1, n_list=n_list, n_probe=n_probe)

    # 3. Build undirected graph
    return build_sym_knn_graph_parallel(indices, n_jobs=n_threads)

#===========================================================================================================

def igraph_label_propagation_from_scann(X, k=10, n_list = 100, n_probe = 10):
    """
    Run label propagation clustering using ScaNN + iGraph.

    Parameters:
    - X: np.ndarray of shape (n, d)
    - k: number of nearest neighbors (default: 10)
    - metric: 'squared_l2' or 'dot_product'

    Returns:
    - labels: list of cluster labels for each point
    """

    n, d = X.shape

    indices, distances = scann_approx_kNN(X, k + 1, n_list, n_probe)

    G = igraph_form_unweighted_sym_KNN_graph(indices)
    labels = run_LPA(G)

    return labels

#===========================================================================================================

def density_peak_eps(X, dc=None, percentile=2.0, top_k=5, plot_decision=False):
    """
    Density Peak Clustering using epsilon (cutoff) distance.

    Parameters:
        X (ndarray): shape (n_samples, n_features), float32
        dc (float): cutoff distance for density (optional)
        percentile (float): used to compute dc if not given
        top_k (int): number of cluster centers to select
        plot_decision (bool): show decision graph

    Returns:
        centers: list of center indices
        labels: array of cluster labels
    """
    n = X.shape[0]
    D = pairwise_distances(X, metric='euclidean')

    # Step 1: Estimate dc if not provided
    if dc is None:
        tri = D[np.triu_indices(n, k=1)]
        dc = np.percentile(tri, percentile)
        print(f"Using dc = {dc:.4f} (percentile = {percentile})")

    # Step 2: Compute density (ε-neighborhood count)
    rho = np.sum(D < dc, axis=1) - 1  # exclude self

    # Step 3: Compute delta (distance to nearest higher density point)
    delta = np.zeros(n)
    nneigh = np.zeros(n, dtype=int)
    sorted_idx = np.argsort(-rho)

    for i, idx in enumerate(sorted_idx):
        if i == 0:
            delta[idx] = np.max(D[idx])
            nneigh[idx] = idx
        else:
            higher = sorted_idx[:i]
            dist_to_higher = D[idx][higher]
            j = np.argmin(dist_to_higher)
            delta[idx] = dist_to_higher[j]
            nneigh[idx] = higher[j]

    # Step 4: Decision graph (optional)
    # if plot_decision:
    #     plt.scatter(rho, delta, s=10)
    #     plt.xlabel("rho (density)")
    #     plt.ylabel("delta (distance to higher density)")
    #     plt.title("DPC Decision Graph")
    #     plt.show()

    # Step 5: Select centers by top rho * delta
    score = rho * delta
    centers = np.argsort(-score)[:top_k]
    print(f"Selected centers: {centers}")

    # Step 6: Label propagation
    labels = -np.ones(n, dtype=int)
    for i, c in enumerate(centers):
        labels[c] = i

    for idx in sorted_idx:
        if labels[idx] == -1:
            labels[idx] = labels[nneigh[idx]]

    return centers, labels

def density_peak_clustering_faiss(X, k=30, dc=None, n_threads=8):
    """
    Density Peak Clustering using FAISS exact kNN search.

    Parameters:
        X: np.ndarray [n_samples, n_features], float32
        k: number of neighbors to use for density estimation
        dc: distance cutoff for Gaussian kernel (optional)

    Returns:
        labels: cluster labels for each point
        rho: local density
        delta: minimum distance to higher density point
    """
    X = X.astype(np.float32)
    n, d = X.shape

    # Step 1: Build FAISS exact index
    faiss.omp_set_num_threads(n_threads) # This is also default
    index = faiss.IndexFlatL2(d)
    index.add(X)

    # Step 2: Find kNN
    distances, neighbors = index.search(X, k + 1)  # includes self at position 0
    distances = distances[:, 1:]  # exclude self
    neighbors = neighbors[:, 1:]

    # Step 3: Compute local density (Gaussian kernel)
    if dc is None:
        dc = np.median(distances)
    rho = np.exp(-(distances ** 2) / (dc ** 2)).sum(axis=1)

    # Step 4: Compute delta (distance to nearest higher-density point)
    delta = np.full(n, np.inf, dtype=np.float32)
    nearest_higher = np.full(n, -1, dtype=np.int32)

    order = np.argsort(-rho)  # descending order of rho

    for i, idx in enumerate(order):
        if i == 0:
            delta[idx] = np.max(distances[idx])  # largest kNN distance for densest point
            continue
        # Search for nearest higher-density point
        for j in neighbors[idx]:
            if rho[j] > rho[idx]:
                d = np.linalg.norm(X[idx] - X[j])
                if d < delta[idx]:
                    delta[idx] = d
                    nearest_higher[idx] = j

    # Step 5: Find cluster centers (e.g., top percentile of rho * delta)
    score = rho * delta
    num_centers = max(2, int(0.01 * n))
    centers = order[np.argsort(-score[:num_centers])]

    # Step 6: Assign cluster labels
    labels = -np.ones(n, dtype=np.int32)
    for i, c in enumerate(centers):
        labels[c] = i

    for idx in order:
        if labels[idx] == -1 and nearest_higher[idx] != -1:
            labels[idx] = labels[nearest_higher[idx]]

    return labels, rho, delta

#===========================================================================================================
def run_sOptics(X, minPts, eps, n_threads=8):

    """
    We test fit_sOptics
    """

    n, d = np.shape(X)
    X = np.transpose(X)


    # Param
    numProj = 1024
    k = 5
    m = 100
    numEmbed = 1024
    sigma = 200  # L2: 2600, L1: 16000 | covtype: L2 200
    dist = "L2"
    clusterNoise = 0
    output = 'sOptics'
    numThreads = n_threads
    verbose = True
    intervalSampling = 0.4
    samplingRatio = 0.02
    seed = -1

    dbs = sDbscan.sDbscan(n, d)
    dbs.set_params(numProj, k, m, dist, numEmbed, sigma, intervalSampling, clusterNoise, samplingRatio, verbose,
                   output, numThreads, seed)

    start = timeit.default_timer()
    dbs.fit_sOptics(X, eps, minPts)
    end = timeit.default_timer()
    print("sOPTICS Time: ", end - start)

    s_reachDist = np.array(dbs.reachability_)
    idx = np.where(s_reachDist < 0)
    s_reachDist[idx] = math.inf
    sOptics = np.take(s_reachDist, np.array(dbs.ordering_))

    # dbs.clear()
    # start = timeit.default_timer()
    # dbs.fit_sngOptics(X, eps, minPts)
    # end = timeit.default_timer()
    # print("sngOPTICS Time: ", end - start)
    #
    # sng_reachDist = np.array(dbs.reachability_)
    # idx = np.where(sng_reachDist < 0)
    # sng_reachDist[idx] = math.inf
    # sngOptics = np.take(sng_reachDist, np.array(dbs.ordering_))

    # Plot two different figure
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(sOptics)
    ax2.plot(sOptics)
    plt.show()

def run_sDbscan(X, minPts, eps, dist = "Cosine", sigma=2600, n_threads=8):

    n, d = np.shape(X)
    X = np.transpose(X)

    # Param
    numProj = 1024
    k = 5
    m = 100

    numEmbed =  1024
    clusterNoise = 0
    numThreads = n_threads
    verbose = False
    intervalSampling = 0.4
    samplingRatio = 0.01
    seed = -1
    output = ""

    dbs = sDbscan.sDbscan(n, d)
    dbs.set_params(numProj, k, m, dist, numEmbed, sigma, intervalSampling, clusterNoise, samplingRatio, verbose,
                   output, numThreads, seed)

    start = timeit.default_timer()
    dbs.fit_sDbscan(X, eps, minPts)
    end = timeit.default_timer()
    print("sDbscan Time: ", end - start)

    return dbs.labels_

def run_sngDbscan(X, minPts, eps, dist = "Cosine", n_threads=8):

    n, d = np.shape(X)
    X = np.transpose(X)

    # Param
    numProj = 1024
    k = 5
    m = 100

    numEmbed = 1024
    sigma = 2600 # L2: 2600
    clusterNoise = 0
    numThreads = n_threads
    verbose = False
    intervalSampling = 0.4
    samplingRatio = 1 # 0.01 for sngDBSCAN, 1 for exact Dbscan with better memory than scikit dbscan
    seed = -1
    output = ""

    dbs = sDbscan.sDbscan(n, d)
    dbs.set_params(numProj, k, m, dist, numEmbed, sigma, intervalSampling, clusterNoise, samplingRatio, verbose,
                   output, numThreads, seed)

    start = timeit.default_timer()
    dbs.fit_sngDbscan(X, eps, minPts)
    end = timeit.default_timer()
    print("sngDbscan Time: ", end - start)

    return dbs.labels_

#===========================================================================================================

if __name__ == '__main__':

    path = "/shared/Dataset/Clustering/"

    # dataset = np.loadtxt(path + 'mnist_all_X')
    # X = np.loadtxt(path + 'mnist_all_X', delimiter=",")
    # X.dtype == np.float32
    # n, d = X.shape


    n = 70000
    d = 784

    X = mmap_bin(path + 'mnist_all_X.bin', n, d)
    X.dtype == np.float32
    X = normalize(X, norm='l2', axis=1)

    # nan_mask = np.isnan(X)
    # print(f"NaN mask: {nan_mask}")
    #
    # nan_indices = np.where(nan_mask)
    # print(f"Indices of NaN values: {nan_indices}")

    true_labels = np.loadtxt(path + 'mnist_all_y_70K_784', dtype=np.int32)
    n_clusters = 10
    n_iter = 20

    """ faiss k-mean """
    # t1 = timeit.default_timer()
    # labels = faiss_kmeans(X, n_clusters)
    # t2 = timeit.default_timer()
    # print('Faiss k-mean Time: {}'.format(t2 - t1))
    #
    # faiss_kmeans_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in faiss_kmeans_ans))

    """ scikit kmean++ """
    # n_iter = 20
    # t1 = timeit.default_timer()
    # kmeans = KMeans(init='random', n_clusters=n_clusters, n_init=4, max_iter=n_iter, random_state=0).fit(X)
    # t2 = timeit.default_timer()
    # print('scikit kmean++ Time: {}'.format(t2 - t1))
    #
    # sci_kmean_ans = getMetric(kmeans.labels_, true_labels)
    # print(' '.join(f"{x:.4f}" for x in sci_kmean_ans))

    """ scikit spectral clustering """
    # gamma = 0.41677414069589885
    # t1 = timeit.default_timer()
    # sc = SpectralClustering(n_clusters=n_clusters, affinity='rbf', gamma=gamma, assign_labels='kmeans', max_iter=n_iter,random_state=0)
    # labels = sc.fit_predict(X)
    # t2 = timeit.default_timer()
    # print('Spectral clustering Time: {}'.format(t2 - t1))
    #
    # spectral_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in spectral_ans))

    """ Nystrom kernel kmean++ """
    # # Compute pairwise Euclidean distances over Subsample to avoid O(n^2) for large MNIST
    # X_sample = X[np.random.choice(len(X), 1000, replace=False)]
    # dists = pairwise_distances(X_sample, metric="euclidean")
    # median_dist = np.median(dists)
    # #
    # # Recommended gamma:
    # gamma = 1 / (2 * median_dist ** 2)
    #
    # n_samples = round(0.01 * n)
    # print("Gamma: ", gamma)
    # print("n_samples: ", n_samples)
    #
    # t1 = timeit.default_timer()
    # labels, Z = nystrom_kernel_kmeans(X, n_clusters=n_clusters, m=n_samples, gamma= gamma, n_iter=n_iter) # gamma = 1/ 2 sigma^2
    # t2 = timeit.default_timer()
    # print('Nystrom kernel k-mean Time: {}'.format(t2 - t1))
    #
    # nys_kmean_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in nys_kmean_ans))

    """ Nystrom spectral clustering """
    # # Compute pairwise Euclidean distances over Subsample to avoid O(n^2) for large MNIST
    # X_sample = X[np.random.choice(len(X), 1000, replace=False)]
    # dists = pairwise_distances(X_sample, metric="euclidean")
    # median_dist = np.median(dists)
    # #
    # # Recommended gamma:
    # gamma = 1 / (2 * median_dist ** 2)
    #
    # n_samples = round(0.01 * n)
    # print("Gamma: ", gamma)
    # print("n_samples: ", n_samples)
    #
    # t1 = timeit.default_timer()
    # labels = nystrom_spectral(X, k=n_clusters, m=n_samples, gamma= gamma, n_iter= n_iter)
    # t2 = timeit.default_timer()
    # print('Nystrom spectral k-mean Time: {}'.format(t2 - t1))
    #
    # nys_spectral_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in nys_spectral_ans))

    """ Faiss LPA """
    # n_neighbors = 12
    # print("Neighbors: ", n_neighbors)
    #
    # t1 = timeit.default_timer()
    # G = build_knn_graph_faiss(X, k=n_neighbors)
    # t2 = timeit.default_timer()
    # print('Faiss Time: {}'.format(t2 - t1))
    # clusters = label_propagation(G) # return [ [1, 3, 5], [2, 4, 6], [10, 11, 7, 8, 9] ], each list is a cluster
    # t2 = timeit.default_timer()
    # print('Faiss LPA Time: {}'.format(t2 - t1))
    #
    # # Build reverse map: point -> cluster ID
    # point_to_cluster = {}
    # for cluster_id, cluster_nodes in enumerate(clusters):
    #     for node in cluster_nodes:
    #         point_to_cluster[node] = cluster_id
    #
    # # Sort by point index to keep order
    # n = len(point_to_cluster)
    # labels = [point_to_cluster[i] for i in range(n)]
    #
    # lpa_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """ Faiss symmetric LPA """
    # n_neighbors_list = [12, 16, 20, 24, 28, 32]
    # for n_neighbors in n_neighbors_list:
    #
    #     print("Neighbors: ", n_neighbors)
    #
    #     t1 = timeit.default_timer()
    #     G = build_symmetric_knn_graph_faiss(X, k=n_neighbors)
    #     t2 = timeit.default_timer()
    #     print('Faiss Time: {}'.format(t2 - t1))
    #     clusters = label_propagation(G) # return [ [1, 3, 5], [2, 4, 6], [10, 11, 7, 8, 9] ], each list is a cluster
    #     t2 = timeit.default_timer()
    #     print('Faiss LPA Time: {}'.format(t2 - t1))
    #
    #     # Build reverse map: point -> cluster ID
    #     point_to_cluster = {}
    #     for cluster_id, cluster_nodes in enumerate(clusters):
    #         for node in cluster_nodes:
    #             point_to_cluster[node] = cluster_id
    #
    #     # Sort by point index to keep order
    #     n = len(point_to_cluster)
    #     labels = [point_to_cluster[i] for i in range(n)]
    #
    #     lpa_ans = getMetric(labels, true_labels)
    #     print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """ Faiss mutual LPA """
    # n_neighbors_list = [12, 16, 20, 24, 28, 32]
    # n_neighbors_list = [50, 100, 150, 200, 250, 300]

    # for n_neighbors in n_neighbors_list:
    #
    #     print("Neighbors: ", n_neighbors)
    #
    #     t1 = timeit.default_timer()
    #     G = build_mutual_knn_graph_faiss(X, k=n_neighbors)
    #     t2 = timeit.default_timer()
    #     print('Faiss Time: {}'.format(t2 - t1))
    #     clusters = label_propagation(G) # return [ [1, 3, 5], [2, 4, 6], [10, 11, 7, 8, 9] ], each list is a cluster
    #     t2 = timeit.default_timer()
    #     print('Faiss LPA Time: {}'.format(t2 - t1))
    #
    #     # Build reverse map: point -> cluster ID
    #     point_to_cluster = {}
    #     for cluster_id, cluster_nodes in enumerate(clusters):
    #         for node in cluster_nodes:
    #             point_to_cluster[node] = cluster_id
    #
    #     # Sort by point index to keep order
    #     n = len(point_to_cluster)
    #     labels = [point_to_cluster[i] for i in range(n)]
    #
    #     lpa_ans = getMetric(labels, true_labels)
    #     print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """ Faiss LPA with n_clusters """
    # n_neighbors = 12
    # print("Neighbors: ", n_neighbors)
    #
    # t1 = timeit.default_timer()
    # G = build_knn_graph_faiss(X, k=n_neighbors)
    # t2 = timeit.default_timer()
    # print('Faiss Time: {}'.format(t2 - t1))
    # labels = label_propagation_k_clusters(G)
    # t2 = timeit.default_timer()
    # print('Faiss LPA with n_cluster Time: {}'.format(t2 - t1))
    #
    # lpa_k_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in lpa_k_ans))

    """ Faiss DPC """


    """ Umap & Hdbscan"""
    # t1 = timeit.default_timer()
    # # 1. Reduce dimensionality, UMAP defaults to n_components=2
    # X_umap = UMAP(n_neighbors=15, min_dist=0.1, metric='cosine').fit_transform(X)
    # # 2. Use HDBSCAN in 2D or 10D
    # labels = HDBSCAN(min_cluster_size=10).fit_predict(X_umap)
    # t2 = timeit.default_timer()
    # print('UMAP & HDBSCAN Time: {}'.format(t2 - t1))
    #
    # umap_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in umap_ans))

    """ Hdbscan """
    # t1 = timeit.default_timer()
    # # Run HDBSCAN directly
    # clusterer = hdbscan.HDBSCAN(
    #     min_cluster_size=30,  # Minimum cluster size
    #     min_samples=None,  # Optional: for noise sensitivity
    #     metric='euclidean',  # Can also use 'manhattan', 'cosine', etc.
    #     cluster_selection_method='eom',  # or 'leaf'
    # )
    #
    # labels = clusterer.fit_predict(X)
    # t2 = timeit.default_timer()
    # print('HDBSCAN Time: {}'.format(t2 - t1))
    # hdbscan_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in hdbscan_ans))

    """ sOptics"""
    # eps = 20000
    # minPts = 12
    # run_sOptics(X, minPts, eps)

    """ sDbscan"""

    # dist = "Cosine"
    # for i in range(5):
    #     minPts_list = [12, 16, 20, 24, 28, 32]
    #     eps_list = [0.1, 0.11, 0.112, 0.13, 0.14, 0.15]
    #     for minPts in minPts_list:
    #         print("minPts: ", minPts)
    #         for eps in eps_list:
    #             ans = run_sDbscan(X, minPts, eps, dist, n_threads=32)
    #             print(' '.join(f"{val:.3f}" for val in ans))
    #
    # dist = "JS"
    # for i in range(5):
    #     minPts_list = [12, 16, 20, 24, 28, 32]
    #     eps_list = [0.1, 0.11, 0.112, 0.13, 0.14, 0.15]
    #     for minPts in minPts_list:
    #         print("minPts: ", minPts)
    #         for eps in eps_list:
    #             ans = run_sDbscan(X, minPts, eps, dist, n_threads=32)
    #             print(' '.join(f"{val:.3f}" for val in ans))
    #
    # dist = "L2"
    # for i in range(5):
    #     minPts_list = [12, 16, 20, 24, 28, 32]
    #     eps_list = [1150, 1200, 1250, 1300, 1350, 1400]
    #     for minPts in minPts_list:
    #         print("minPts: ", minPts)
    #         for eps in eps_list:
    #             ans = run_sDbscan(X, minPts, eps, dist, sigma=2600, n_threads=32)
    #             print(' '.join(f"{val:.3f}" for val in ans))


    # dist = "L1"
    # for i in range(1):
    #     minPts_list = [4, 6, 8, 10, 12, 14]
    #     eps_list = [100, 110, 120, 130, 140, 150]
    #     for minPts in minPts_list:
    #         print("minPts: ", minPts)
    #         for eps in eps_list:
    #             ans = run_sDbscan(X, minPts, eps, dist, sigma=16000, n_threads=8)
    #             print(' '.join(f"{val:.3f}" for val in ans))

    # run_sDbscan(X, minPts=24, eps=0.13, dist="Cosine", n_threads = 32)

    """ sngDbscan"""
    # dist = "JS"
    # for i in range(5):
    #     minPts_list = [12, 16, 20, 24, 28, 32]
    #     eps_list = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
    #     for minPts in minPts_list:
    #         print("minPts: ", minPts)
    #         for eps in eps_list:
    #             ans = run_sngDbscan(X, minPts, eps, dist)
    #             print(' '.join(f"{val:.3f}" for val in ans))

    # run_sngDbscan(X, minPts=24, eps=0.13, dist="Cosine", n_threads = 32)

    """ Dbscan via sngDbscan """
    # dist = "JS"
    # for i in range(1):
    #     minPts_list = [12, 16, 20, 24, 28, 32]
    #     eps_list = [0.11]
    #     for minPts in minPts_list:
    #         print("minPts: ", minPts)
    #         for eps in eps_list:
    #             ans = run_sngDbscan(X, minPts, eps, dist)
    #             print(' '.join(f"{val:.3f}" for val in ans))

    """ Optics Xi - Not test """
    # n_neighbors = 24
    # t1 = timeit.default_timer()
    # optics_model = OPTICS(min_samples=n_neighbors, xi=0.05, min_cluster_size=0.05)
    # optics_model.fit(X)
    # t2 = timeit.default_timer()
    # print('Optics Xi Time: {}'.format(t2 - t1))
    #
    # optics_xi_ans = getMetric(optics_model.labels_, true_labels)
    # print(' '.join(f"{x:.4f}" for x in optics_xi_ans))
    #
    # reachability = optics_model.reachability_
    # ordering = optics_model.ordering_
    # space = np.arange(len(X))
    # plt.figure(figsize=(10, 5))
    # plt.plot(space, reachability[ordering], 'k-', label='Reachability')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Reachability Distance')
    # plt.title('OPTICS Reachability Plot')
    # plt.grid(True)
    # plt.show()
    #
    # # Extract DBSCAN-style clustering at different eps values
    # eps_values = [0.3, 0.5, 0.7]
    # for eps in eps_values:
    #     labels = cluster_optics_dbscan(
    #         reachability=optics_model.reachability_,
    #         core_distances=optics_model.core_distances_,
    #         ordering=optics_model.ordering_,
    #         eps=eps
    #     )
    #     dbscan_ans = getMetric(labels, true_labels)
    #     print(' '.join(f"{x:.4f}" for x in dbscan_ans))

