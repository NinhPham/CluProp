import os
# must set before import
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["FAISS_NUM_THREADS"] = "8"

import faiss
import hdbscan
import numpy as np
import math
from sklearn.cluster import DBSCAN, OPTICS, KMeans, SpectralClustering,cluster_optics_dbscan
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.metrics.cluster import pair_confusion_matrix

from sklearn.neighbors import NearestNeighbors
import networkx as nx
import random
from collections import Counter

from umap import UMAP
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt

import timeit
import gc
from concurrent.futures import ThreadPoolExecutor

def mmap_bin(bin_path, num_rows, num_cols, dtype=np.float32):
    return np.memmap(bin_path, dtype=dtype, mode='r', shape=(num_rows, num_cols))

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

def faiss_kmeans(X, n_clusters=10, n_iter=20, gpu=False):
    """
    X: numpy array [n_samples, n_features]
    n_clusters: number of clusters
    n_iter: number of iterations
    gpu: whether to use GPU (True/False)
    """
    X = X.astype(np.float32)
    d = X.shape[1]  # dimensionality

    kmeans = faiss.Kmeans(
        d=d,
        k=n_clusters,
        niter=n_iter,
        nredo=1,
        verbose=True,
        min_points_per_centroid=1,  # avoid dropping centroids
        max_points_per_centroid=1000000  # disable sampling cap
    )
    import multiprocessing
    # faiss.omp_set_num_threads(multiprocessing.cpu_count()) # This is also default
    # faiss.omp_set_num_threads(16) # This is also default
    kmeans.train(X.astype(np.float32))

    # labeling by 1NN
    distances, labels = kmeans.index.search(X, 1)
    labels = labels.flatten()

    return labels

def nystrom_kernel_kmeans(X, n_clusters=10, m=1000, gamma=0.5, n_iter=10, n_job = 16):
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
    W_mm = rbf_kernel(X_landmarks, X_landmarks, gamma=gamma)  # [m x m]
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

def build_knn_graph_faiss(X, k=10):
    """
    Build an undirected k-NN graph using FAISS for neighbor search.
    X must be a float32 NumPy array.
    """
    X = X.astype(np.float32)
    n, d = X.shape

    # 1. Create FAISS index
    import multiprocessing
    # faiss.omp_set_num_threads(multiprocessing.cpu_count()) # This is also default
    index = faiss.IndexFlatL2(d)  # L2 distance (Euclidean)
    index.add(X)

    # 2. Perform search (note: k+1 because the closest is the point itself)
    distances, indices = index.search(X, k + 1)

    # 3. Build undirected graph
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in indices[i][1:]:  # skip self-match
            G.add_edge(i, j)

    return G

def label_propagation(G, max_iter=100):
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
    return list(clusters.values())

def label_propagation_k_clusters(G, k=10, max_iter=100, seed_strategy='degree'):
    n = G.number_of_nodes()
    labels = {node: -1 for node in G.nodes()}

    # Step 1: Choose k seed nodes
    if seed_strategy == 'random':
        seeds = random.sample(list(G.nodes()), k)
    elif seed_strategy == 'degree':
        seeds = sorted(G.degree(), key=lambda x: -x[1])
        seeds = [node for node, _ in seeds[:k]]
    else:
        raise ValueError("Unknown seed strategy")

    # Step 2: Assign unique labels to seeds
    for i, node in enumerate(seeds):
        labels[node] = i

    # Step 3: Propagate labels
    for _ in range(max_iter):
        changed = False
        nodes = list(G.nodes())
        random.shuffle(nodes)

        for node in nodes:
            if labels[node] != -1:
                continue  # already labeled

            neighbor_labels = [labels[neigh] for neigh in G.neighbors(node) if labels[neigh] != -1]
            if neighbor_labels:
                new_label = Counter(neighbor_labels).most_common(1)[0][0]
                labels[node] = new_label
                changed = True

        if not changed:
            break  # convergence

    # Assign remaining unlabeled nodes (if any)
    for node in labels:
        if labels[node] == -1:
            neighbor_labels = [labels[neigh] for neigh in G.neighbors(node) if labels[neigh] != -1]
            if neighbor_labels:
                labels[node] = Counter(neighbor_labels).most_common(1)[0][0]
            else:
                labels[node] = random.randint(0, k-1)

    return [labels[i] for i in range(n)]

def density_peak_clustering_faiss(X, k=30, dc=None):
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

if __name__ == '__main__':

    path = "/shared/Dataset/Clustering/"
    bin_file = path + 'covtype_X.bin'

    n = 581012
    d = 54

    X = mmap_bin(bin_file, n, d)
    X = normalize(X, norm='l2', axis=1)

    true_labels = np.loadtxt(path + 'covtype_y_581012_54')
    n_clusters = 7

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
    # n_clusters = 10
    # n_iter = 20
    #
    # # sigma = 2600  # mnist: sigma = 2600 for L2
    # # gamma = 1 / (2 * sigma * sigma)
    # print("Gamma: ", gamma)
    #
    # t1 = timeit.default_timer()
    # labels, Z = nystrom_kernel_kmeans(X, n_clusters=n_clusters, m=n_samples, gamma= gamma, n_iter=n_iter) # gamma = 1/ 2 sigma^2
    # t2 = timeit.default_timer()
    # print('Nystrom kernel k-mean Time: {}'.format(t2 - t1))
    #
    # nys_kmean_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in nys_kmean_ans))

    """ Nystrom spectral clustering """
    # n_samples = round(0.01 * n)
    # n_clusters = 10
    # sigma = 2600  # mnist: sigma = 2600 for L2
    # n_iter = 20
    # gamma = 0.41677414069589885
    #
    # t1 = timeit.default_timer()
    # labels = nystrom_spectral(X, k=n_clusters, m=n_samples, gamma= gamma, n_iter= n_iter)
    # t2 = timeit.default_timer()
    # print('Nystrom spectral k-mean Time: {}'.format(t2 - t1))
    #
    # nys_spectral_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in nys_spectral_ans))

    """ Faiss LPA """
    n_neighbors = 20
    print("Neighbors: ", n_neighbors)

    t1 = timeit.default_timer()
    G = build_knn_graph_faiss(X, k=n_neighbors)
    t2 = timeit.default_timer()
    print('Faiss Time: {}'.format(t2 - t1))
    clusters = label_propagation(G) # return [ [1, 3, 5], [2, 4, 6], [10, 11, 7, 8, 9] ], each list is a cluster
    t2 = timeit.default_timer()
    print('Faiss LPA Time: {}'.format(t2 - t1))

    # Build reverse map: point -> cluster ID
    point_to_cluster = {}
    for cluster_id, cluster_nodes in enumerate(clusters):
        for node in cluster_nodes:
            point_to_cluster[node] = cluster_id

    # Sort by point index to keep order
    n = len(point_to_cluster)
    labels = [point_to_cluster[i] for i in range(n)]

    lpa_ans = getMetric(labels, true_labels)
    print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """ Faiss LPA with n_clusters """
    # n_neighbors = 28
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

    """ DPC """

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

