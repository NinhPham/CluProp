import os


import numpy as np
import math
from sklearn.cluster import DBSCAN, OPTICS, KMeans, SpectralClustering, cluster_optics_dbscan
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.metrics.cluster import pair_confusion_matrix
from sklearn.neighbors import NearestNeighbors
from pynndescent import NNDescent
import timeit
import hdbscan
import cluprop

from scipy.sparse.linalg import eigsh
from scipy import sparse
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans

import utils
from utils import getMetric

# --------------------------------------------------
# SpecACl fit_predict
# Based on:
#   1) build adjacency matrix W
#   2) top-d eigenvectors/eigenvalues of W
#   3) projected embedding U_jk = |V_jk| * sqrt(|lambda_k|)
#   4) k-means on U
# --------------------------------------------------
def spectacl_fit_predict(X, metric, n_clusters, n_neighbors=10, n_components=None):
    n = X.shape[0]

    if n_components is None:
        n_components = n_clusters

    # Build sparse kNN adjacency graph W
    W = kneighbors_graph(
        X,
        metric = metric,
        n_neighbors=n_neighbors,
        mode="connectivity",
        include_self=False,
        n_jobs=8
    )

    # Make graph symmetric
    W = 0.5 * (W + W.T)

    # Largest algebraic eigenvalues/eigenvectors of adjacency matrix
    vals, vecs = eigsh(W.astype(float), k=n_components, which="LA")

    # Sort descending
    order = np.argsort(-vals)
    vals = vals[order]
    vecs = vecs[:, order]

    # SpecACl projected embedding
    U = np.abs(vecs) * np.sqrt(np.abs(vals))

    # k-means on projected embedding
    labels = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=42
    ).fit_predict(U)

    return labels

def dpc_fit_predict(X, metric, n_clusters, dc_percent=2.0):
    n = X.shape[0]
    D = pairwise_distances(X, metric=metric)

    triu = D[np.triu_indices(n, k=1)]
    dc = np.percentile(triu, dc_percent)

    rho = np.sum(np.exp(-(D / dc) ** 2), axis=1) - 1.0
    order = np.argsort(-rho)

    delta = np.zeros(n)
    nneigh = np.full(n, -1, dtype=int)

    delta[order[0]] = np.max(D[order[0]])

    for k in range(1, n):
        i = order[k]
        higher = order[:k]
        j = higher[np.argmin(D[i, higher])]
        delta[i] = D[i, j]
        nneigh[i] = j

    gamma = rho * delta
    centers = np.argsort(-gamma)[:n_clusters]

    labels = np.full(n, -1, dtype=int)
    for c, idx in enumerate(centers):
        labels[idx] = c

    for i in order:
        if labels[i] == -1:
            labels[i] = labels[nneigh[i]]

    return labels


if __name__ == '__main__':


    # path = "/Users/pham/Dropbox-UniofAuckland/Ninh Pham/Working/_Code/Matlab/USPEC/"  # Mac
    # from scipy.io import loadmat
    # path = "/home/npha145/Uni of Auckland Dropbox/Ninh Pham/Working/_Code/Matlab/USPEC/"
    # dataName = "MNIST"
    # mat = loadmat(f"{path}data_{dataName}.mat")
    # X = mat["fea"]
    # y = mat["gt"].ravel()

    # path = "/Users/pham/Dropbox-UniofAuckland/Ninh Pham/Working/_Code/C++/CluProp/test/Dataset/"  # Mac
    path = "/home/npha145/Uni of Auckland Dropbox/Ninh Pham/Working/_Code/C++/CluProp/test/Dataset/"
    dataName = "soybean" # multiple-features, optdigits, pendigits, usps, semeion, letter, dermatology, soybean
    X = np.loadtxt(f"{path}{dataName}-data.txt", delimiter=",")
    y = np.loadtxt(f"{path}{dataName}-labels.txt", delimiter=",")


    print("X shape:", X.shape)
    n_clusters = len(np.unique(y))
    print("Number of clusters: ", n_clusters)

    n, d = X.shape
    n_threads = 8
    distance = "euclidean"

    # If cosine, then call this function
    # X = normalize(X, norm='l2', axis=1)

    """ CluProp """

    # Step 1: Compute exact kNN
    n_threads = 8
    k_max = 50
    n_repeats = 1
    k_expand = 1
    # Exact L2
    indices, distances = utils.faiss_kNN(X, k=k_max + 1, n_threads=8) # On MAC only
    indices = indices.astype(np.int32)
    distances = distances.astype(np.float32)

    # Step 2: Leiden / Louvain / LPA / DANE
    n_neighbors_list = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    # n_neighbors_list = [4, 6, 8, 10, 12, 14]
    # n_neighbors_list = [20, 25, 30, 35, 40, 45, 50]
    # n_neighbors_list = [60, 70, 80, 90, 100]
    # n_neighbors_list = [1, 2, 3, 4, 5, 6]

    for n_neighbors in n_neighbors_list:

        print('n_neighbors: ', n_neighbors)
        K = min(n_neighbors + 1, k_max)  # +1 for faiss
        k_expand = round(K / 1)

        # LPA: need + 1 for Faiss
        unweighted_graph = utils.fast_unweighted_sym_knng_igraph(indices[:, 1 : K], verbose=False)

        for i in range(n_repeats):

            t1 = timeit.default_timer()
            labels = utils.run_LPA(unweighted_graph)
            t2 = timeit.default_timer()
            # print('LPA Time: {}'.format(t2 - t1))
            lpa_ans = getMetric(labels, y)
            print(' '.join(f"{x:.4f}" for x in lpa_ans))

        # Note: exp_weight=False gives slightly higher accuracy, need + 1 for Faiss
        # Leiden
        # This is G_k
        t1 = timeit.default_timer()
        weighted_graph = utils.fast_weighted_sym_knng_igraph(indices[:, 1 : K], distances[:, 1 : K], use_exp_weight=False,verbose=False)
        # weighted_graph = utils.fast_local_scaled_sym_knng_igraph(indices[:, 1: K], distances[:, 1: K], verbose=False)
        t2 = timeit.default_timer()
        # print('Graph Construction Time: {}'.format(t2 - t1))

        # Mutual G_k
        # weighted_graph = utils.fast_weighted_mutual_knng_igraph(indices[:, 1 : K], distances[:, 1 : K], use_exp_weight=False,verbose=False)

        for i in range(n_repeats):

            t1 = timeit.default_timer()
            labels = utils.run_leiden(weighted_graph)
            t2 = timeit.default_timer()
            # print('Leiden Time: {}'.format(t2 - t1))
            lpa_ans = getMetric(labels, y)
            print(' '.join(f"{x:.4f}" for x in lpa_ans))

        # Louvain
        # This is G_k
        # weighted_graph = utils.fast_weighted_sym_knng_igraph(indices[:, 1 : K], distances[:, 1 : K], use_exp_weight=False,verbose=False)

        # for i in range(n_repeats):
        #
        #     t1 = timeit.default_timer()
        #     labels = utils.run_louvain(weighted_graph)
        #     t2 = timeit.default_timer()
        #     print('Louvain Time: {}'.format(t2 - t1))
        #     lpa_ans = getMetric(labels, y)
        #     print(' '.join(f"{x:.4f}" for x in lpa_ans))

        # Step 3: DANE
        dbs = cluprop.cluprop(n, d)
        # dbs.set_min_cluster_size(50)

        for i in range(n_repeats):

            t1 = timeit.default_timer()
            dbs.knn_dane(indices[:,  1 : K], distances[:,  1 : K], K, k_expand)
            t2 = timeit.default_timer()
            # print('DANE Time: {}'.format(t2 - t1))
            lpa_ans = getMetric(np.array(dbs.labels_), y)
            print(' '.join(f"{x:.4f}" for x in lpa_ans))

    # exit()

    """DBSCAN"""
    eps_list = np.arange(0.05, 0.81, 0.05)
    minPts = 10

    all_labels = {}
    summary = []

    for eps in eps_list:
        model = DBSCAN(eps=float(eps), min_samples=minPts, metric=distance, n_jobs=8)
        labels = model.fit_predict(X)
        all_labels[eps] = labels

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = np.mean(labels == -1)
        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)
        ami = adjusted_mutual_info_score(y, labels)

        summary.append([eps, n_clusters_, noise_ratio, ami, ari, nmi])

    print(" === DBSCAN === ")
    print(" eps    n_clusters_   noise_ratio  AMI     ARI     NMI")
    for row in summary:
        print(f"{row[0]:4.2f} {row[1]:10d} {row[2]:12.4f} {row[3]:8.4f} {row[4]:8.4f} {row[5]:8.4f}")

    """ OPTICS """
    minPts = 10
    optics_model = OPTICS(
        min_samples=minPts,
        metric=distance,
        cluster_method="xi",
        xi=0.05,
        n_jobs=8
    )
    optics_model.fit(X)

    eps_list = np.arange(0.05, 0.81, 0.05)
    results = []

    for eps in eps_list:
        labels = cluster_optics_dbscan(
            reachability=optics_model.reachability_,
            core_distances=optics_model.core_distances_,
            ordering=optics_model.ordering_,
            eps=eps
        )

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        noise_ratio = np.mean(labels == -1)

        ami = adjusted_mutual_info_score(y, labels)
        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)

        results.append((eps, n_clusters_, noise_ratio, ami, ari, nmi))

    print(" === OPTICS === ")
    print(" eps    n_clusters_   noise_ratio     AMI    ARI    NMI")
    for row in results:
        print(f"{row[0]:4.2f} {row[1]:10d} {row[2]:12.4f} {row[3]:8.4f} {row[4]:8.4f} {row[5]:8.4f} ")


    """ Spectral clustering """
    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity="rbf",
        gamma=1.0,
        assign_labels="kmeans",
        random_state=42
    )

    labels = model.fit_predict(X)

    ami = adjusted_mutual_info_score(y, labels)
    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)

    print(" === Spectral Clustering === ")
    print("Number of clusters:", len(set(labels)))
    print("AMI:", ami)
    print("ARI:", ari)
    print("NMI:", nmi)

    """ SpecACl"""

    neighbor_list = [10, 20, 30, 50, 80]
    results = []

    for n_neighbors in neighbor_list:
        labels = spectacl_fit_predict(
            X,
            metric = distance,
            n_clusters=n_clusters,
            n_neighbors=n_neighbors,
            n_components=n_clusters
        )
        ami = adjusted_mutual_info_score(y, labels)
        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)

        results.append((n_neighbors, n_clusters, ami, ari, nmi))

    print(" === SpecACl === ")
    print(" k    n_clusters    AMI   ARI   NMI")
    for row in results:
        print(f"{row[0]:4.2f} {row[1]:10d} {row[2]:12.4f} {row[3]:8.4f} {row[4]:8.4f}")


    """ Hdbscan (not support multi-threading) """
    min_cluster_size_list = [5, 10, 20, 30, 50]
    results = []

    for min_cluster_size in min_cluster_size_list:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=None,          # default: same spirit as HDBSCAN default behavior
            metric=distance,
            cluster_selection_method="eom"
        )

        labels = clusterer.fit_predict(X)

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)
        noise_ratio = np.mean(labels == -1)

        ami = adjusted_mutual_info_score(y, labels)
        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)

        results.append(
            (min_cluster_size, n_clusters_, noise_ratio, ami, ari, nmi)
        )

    print(" === HDBSCAN === ")
    print("min_size   n_clusters_   noise_ratio    AMI   ARI    NMI")
    for row in results:
        print(f"{row[0]:4.2f} {row[1]:10d} {row[2]:12.4f} {row[3]:8.4f} {row[4]:8.4f} {row[5]:8.4f}")

    """ DPC"""
    dc_percent_list = [0.5, 1.0, 2.0, 3.0, 5.0]
    results = []

    for dc_percent in dc_percent_list:
        labels = dpc_fit_predict(X, metric=distance, n_clusters=n_clusters, dc_percent=dc_percent)

        ami = adjusted_mutual_info_score(y, labels)
        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)

        results.append((dc_percent, len(np.unique(labels)), ami, ari, nmi))

    print(" === DPC === ")
    print("dc%   n_clusters_  AMI  ARI NMI")
    for row in results:
        print(f"{row[0]:4.2f} {row[1]:10d} {row[2]:12.4f} {row[3]:8.4f} {row[4]:8.4f}")

    """ DPA (new alg for DPC) but I can only install this package dadapy on Mac Book """
    from dadapy import Data

    # Clear duplicates, otherwise cause bugs
    _, unique_indices = np.unique(X, axis=0, return_index=True)

    unique_indices = np.sort(unique_indices)

    X = X[unique_indices]
    y = y[unique_indices]
    # --------------------------------------------------
    # Build DADApy object
    # maxk should be large enough for neighborhood-based estimates
    # --------------------------------------------------
    data = Data(X, maxk=100)

    # Compute pairwise neighbor structure / densities
    data.compute_distances()
    data.compute_id_2NN()
    data.compute_density_kNN()

    # --------------------------------------------------
    # Run ADP / DPA clustering
    # Z controls statistical confidence of detected peaks
    # --------------------------------------------------
    clusters = data.compute_clustering_ADP(Z=1.65, halo=False)

    # Depending on version, cluster labels may be stored here:
    labels = np.array(data.cluster_assignment)

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    ami = adjusted_mutual_info_score(y, labels)
    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)

    n_clusters = len(np.unique(labels))

    print(" === DPA === ")
    print("Number of clusters:", n_clusters)
    print("AMI:", ami)
    print("ARI:", ari)
    print("NMI:", nmi)

