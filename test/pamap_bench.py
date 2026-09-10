import os

import utils
from utils import getMetric

import cluprop
import faiss

import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from sklearn.preprocessing import normalize

from sklearn.neighbors import NearestNeighbors # Used for exact L1
from pynndescent import NNDescent

import timeit

from pathlib import Path
from scipy.io import loadmat

if __name__ == '__main__':

    path = Path("~/Work/Datasets/Clustering/").expanduser()
    savePath = path / "pamap2_output"

    n = 1770131
    d = 51
    bin_file = path / 'pamap2_X_no_0.bin'

    X = utils.mmap_bin(path / 'pamap2_X_no_0.bin', n, d)
    X.dtype == np.float32

    ## Check nan
    # nan_mask = np.isnan(X)
    # print(f"NaN mask: {nan_mask}")
    # nan_indices = np.where(nan_mask)
    # print(f"Indices of NaN values: {nan_indices}")

    true_labels = np.loadtxt(path / 'pamap2_y_no_0_1770131_51', dtype=np.int32)

    n_clusters = 18
    n_iter = 100
    n_repeats = 1
    n_threads = 8

    """====================="""

    """ Compute exact kNN """
    # n_threads = 32
    # k_max = 200
    #
    # # Exact L2
    # indices, distances = utils.faiss_kNN(X, k=k_max + 1, n_threads=n_threads)
    # np.save(savePath / "exact_L2_200_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath / "exact_L2_200_distances.npy", distances)  # shape: (n, k), dtype: float32
    #
    # # Exact L1
    # nbrs = NearestNeighbors(n_neighbors=k_max + 1, metric='manhattan',n_jobs=n_threads).fit(X)
    # distances, indices = nbrs.kneighbors(X)
    # np.save(savePath / "exact_L1_200_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath / "exact_L1_200_distances.npy", distances)  # shape: (n, k), dtype: float32
    #
    # # Exact Cosine
    # X = normalize(X, norm='l2', axis=1)
    # indices, distances = utils.faiss_kNN(X, k=k_max + 1, n_threads=n_threads)
    # np.save(savePath / "exact_Cosine_200_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath / "exact_Cosine_200_distances.npy", distances)  # shape: (n, k), dtype: float32

    """ Compute Faiss approx kNN (IVF and IVFPQ) """
    # n_threads = 8
    # k_max = 500
    #
    # # Faiss params
    # nlist = 512
    # nprobe = 10
    # m = 3
    #
    # # L2
    # indices, distances = utils.faiss_approx_kNN_IVF(X, k=k_max + 1, n_list = nlist, n_probe = nprobe, n_threads=n_threads)
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    # np.save(savePath / "ivf_512_10_L2_500_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath / "ivf_512_10_L2_500_distances.npy", distances)  # shape: (n, k), dtype: float32
    #
    # indices, distances = utils.faiss_approx_kNN_IVFPQ(X, k=k_max + 1, n_subquantizer=m, n_list = nlist, n_probe = nprobe, n_threads=n_threads)
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    # np.save(savePath / "ivfpq_512_10_3_L2_500_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath / "ivfpq_512_10_3_L2_500_distances.npy", distances)  # shape: (n, k), dtype: float32
    #
    # # Cosine
    # X = normalize(X, norm='l2', axis=1)
    #
    # indices, distances = utils.faiss_approx_kNN_IVF(X, k=k_max + 1, n_list = nlist, n_probe = nprobe, n_threads=n_threads)
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    # np.save(savePath / "ivf_512_10_Cosine_500_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath / "ivf_512_10_Cosine_500_distances.npy", distances)  # shape: (n, k), dtype: float32
    #
    # indices, distances = utils.faiss_approx_kNN_IVFPQ(X, k=k_max + 1, n_subquantizer=m, n_list = nlist, n_probe = nprobe, n_threads=n_threads)
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    # np.save(savePath / "ivfpq_512_10_3_Cosine_500_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath / "ivfpq_512_10_3_Cosine_500_distances.npy", distances)  # shape: (n, k), dtype: float32

    """ Compute NNDescent """
    # n_threads = 8
    # k_max = 50
    # seed = 42
    # dist="euclidean"
    # #
    # # # NNDescent params
    # n_trees = 8
    # n_iters = 1
    # # leafSize = int(k / n_trees)
    # leafSize = 50
    #
    # t1 = timeit.default_timer()
    #
    # # Warmup njit
    # # NNDescent(X, n_neighbors=k_max, random_state=seed, tree_init=True,
    # #           # init_graph=indices,
    # #           metric=dist, n_iters=1, n_jobs=n_threads)
    #
    # # It does not count the point itself
    # indices, distances = NNDescent(X, n_neighbors=k_max, random_state=None,
    #                                n_trees=n_trees,          # <-- number of RP trees (you choose)
    #                                leaf_size=leafSize,        # good rule: ≈ n_neighbors
    #                                metric=dist, n_iters=n_iters, n_jobs=n_threads).neighbor_graph
    #
    # build_time = timeit.default_timer() - t1
    #
    # print(f"Beelink-8: RPT: metric={dist} n_trees={n_trees:2d} n_iters={n_iters:2d} leafSize={leafSize:2d} k_max={k_max:2d} time={build_time:.4f}s")
    #
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    #
    # np.save(savePath / f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath / f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32


    """====================="""

    """ iGraph propagation with precomputed Faiss/NNDescent symmetric kNN (need +1 as Faiss consider the point itself as part of kNN) """
    # n_threads = 8
    # indices = np.load(savePath / "ivfpq_512_10_3_L2_500_indices.npy")    # shape: (n, k), dtype: int64
    # distances = np.load(savePath / "ivfpq_512_10_3_L2_500_distances.npy")  # shape: (n, k), dtype: float32
    # indices = np.load(savePath / "ivf_512_10_L2_500_indices.npy")    # shape: (n, k), dtype: int64
    # distances = np.load(savePath / "ivf_512_10_L2_500_distances.npy")  # shape: (n, k), dtype: float32

    # NNDescent params
    # n_iters = 1
    # n_trees = 8
    # k_max = 50
    # leafSize = 50
    # dist = "euclidean"
    #
    # indices = np.load(savePath / f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_indices.npy")    # shape: (n, k), dtype: int64
    # distances = np.load(savePath / f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_distances.npy")  # shape: (n, k), dtype: float32
    #
    # # n_neighbors_list = [10, 12, 14, 16, 18, 20]
    # n_neighbors_list = [12, 16, 20, 24, 28, 32]
    #
    # #
    # print(n_neighbors_list)
    #
    # for n_neighbors in n_neighbors_list:
    #
    #     print('n_neighbors: ', n_neighbors)
    #     K = min(n_neighbors + 1, k_max) # Faiss, NNDescent: + 1
    #
    #     # unweighted_graph = utils.fast_unweighted_sym_knng_igraph(indices[:, 1 : K], verbose=False)
    #     #
    #     # for i in range(n_repeats):
    #     #
    #     #     t1 = timeit.default_timer()
    #     #     labels = utils.run_LPA(unweighted_graph)
    #     #     t2 = timeit.default_timer()
    #     #     print('LPA Time: {}'.format(t2 - t1))
    #     #     lpa_ans = getMetric(labels, true_labels)
    #     #     print(' '.join(f"{x:.4f}" for x in lpa_ans))
    #
    #     # # Note: exp_weight=False gives slightly higher accuracy, need + 1 for Faiss
    #     # # Leiden
    #     t1 = timeit.default_timer()
    #     weighted_graph = utils.fast_weighted_sym_knng_igraph(indices[:, 1 : K], distances[:, 1 : K], use_exp_weight=False,verbose=False)
    #     t2 = timeit.default_timer()
    #     print('Graph Construction Time: {}'.format(t2 - t1))
    #
    #     # Mutual G_k
    #     # weighted_graph = utils.fast_weighted_mutual_knng_igraph(indices[:, 1 : K], distances[:, 1 : K], use_exp_weight=False,verbose=False)
    #
    #     for i in range(n_repeats):
    #
    #         t1 = timeit.default_timer()
    #         labels = utils.run_leiden(weighted_graph)
    #         t2 = timeit.default_timer()
    #         print('Leiden Time: {}'.format(t2 - t1))
    #         lpa_ans = getMetric(labels, true_labels)
    #         print(' '.join(f"{x:.4f}" for x in lpa_ans))
    #
    #     # # Louvain
    #     # # This is G_k
    #     # # weighted_graph = utils.fast_weighted_sym_knng_igraph(indices[:, :n_neighbors], distances[:, :n_neighbors], use_exp_weight=False,verbose=False)
    #     #
    #     # for i in range(n_repeats):
    #     #
    #     #     t1 = timeit.default_timer()
    #     #     labels = utils.run_louvain(weighted_graph)
    #     #     t2 = timeit.default_timer()
    #     #     print('Louvain Time: {}'.format(t2 - t1))
    #     #     lpa_ans = getMetric(labels, true_labels)
    #     #     print(' '.join(f"{x:.4f}" for x in lpa_ans))
    #
    #     # DANE
    #     t1 = timeit.default_timer()
    #     dbs = cluprop.cluprop()
    #     dbs.n_threads = n_threads
    #
    #     dbs.knn_dane(indices[:, 1 : K], distances[:, 1 : K], n_neighbors)
    #     t2 = timeit.default_timer()
    #     print('DANE Time: {}'.format(t2 - t1))
    #     lpa_ans = getMetric(np.array(dbs.labels_), true_labels)
    #     print(' '.join(f"{x:.4f}" for x in lpa_ans))


    """====================="""
    """ Test param k_expand for DANE where k_expand > k"""
    n_threads = 8
    n_trees = 8
    n_iters = 1
    k_max = 50
    leafSize = 50
    dist="euclidean"

    indices = np.load(savePath / f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_indices.npy")    # shape: (n, k), dtype: int64
    distances = np.load(savePath / f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_distances.npy")  # shape: (n, k), dtype: float32

    # print('shape of array :', indices.shape)
    # print(indices[0, 0 : 10])
    # print(distances[0, 0 : 10])

    dbs = cluprop.cluprop()
    dbs.n_threads = n_threads

    n_neighbors_list = [5, 10, 15, 20, 25, 30]

    # this param will significantly increase number of clusters (i.e. identifying more noise clusters)
    # dbs.set_propagation_cutoff()

    print(n_neighbors_list)

    for n_neighbors in n_neighbors_list:

        print('n_neighbors: ', n_neighbors) # k' in the paper

        # clupig
        t1 = timeit.default_timer()

        # G_K where K = ck
        # NNDescent and Faiss consider the point itself, so need to start from 1
        K = min(n_neighbors + 1, k_max)
        dbs.knn_dane(indices[:, 1 : K], distances[:, 1 : K], n_neighbors)
        t2 = timeit.default_timer()
        print('DANE Time: {}'.format(t2 - t1))
        lpa_ans = getMetric(np.array(dbs.labels_), true_labels)
        print(' '.join(f"{x:.4f}" for x in lpa_ans))


        # G_kmax where ck <= K_max,
        K = k_max
        k_expand = min(2 * n_neighbors, k_max)
        t1 = timeit.default_timer()
        dbs.knn_dane(indices[:, 1 : K], distances[:, 1 : K], n_neighbors, k_expand)
        t2 = timeit.default_timer()
        print('DANE Time with k_expand: {}'.format(t2 - t1))
        lpa_ans = getMetric(np.array(dbs.labels_), true_labels)
        print(' '.join(f"{x:.4f}" for x in lpa_ans))



    """ Hdbscan """
    ## Note: HDBSCAN is very slow - it took 1 hour on Mnist 70K points
    # import hdbscan
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
    # eps = 50
    # dist = "L2"
    # sigma = 40
    # minPts = 50
    # n_threads = 8
    # k = 10
    # m = 100
    # utils.run_sOptics(X, minPts, eps, dist, k, m, sigma, n_threads)

    """ sDbscan"""
    # dist = "L2"
    # sigma = 40
    # k = 10
    # m = 100
    # print("sigma: ", sigma)
    # n_threads = 8
    # for i in range(1):
    #     minPts_list = [50]
    #     eps_list = [6, 9, 12, 15, 18, 21]
    #     for minPts in minPts_list:
    #         print("minPts: ", minPts)
    #         for eps in eps_list:
    #             labels = utils.run_sDbscan(X, minPts, eps, dist, k, m, sigma, n_threads)
    #             ans = getMetric(labels, true_labels)
    #             print(' '.join(f"{x:.4f}" for x in ans))
    # utils.run_sDbscan(X, minPts=50, eps=18, dist, sigma, n_threads = 8)

    """ sngDbscan"""
    # dist = "L2"
    # n_threads = 8
    # p = 0.01
    # for i in range(1):
    #     minPts_list = [50]
    #     eps_list = [6, 9, 12, 15, 18, 21]
    #     for minPts in minPts_list:
    #         print("minPts: ", minPts)
    #         for eps in eps_list:
    #             labels = utils.run_sngDbscan(X, minPts, eps, dist, p, n_threads)
    #             ans = getMetric(labels, true_labels)
    #             print(' '.join(f"{x:.4f}" for x in ans))
    # utils.run_sngDbscan(X, minPts=50, eps=15, dist="L2", p, n_threads)

    """====================="""

    """ faiss k-mean """
    # n_threads = 8
    # n_iter = 20
    #
    # t1 = timeit.default_timer()
    # labels = utils.faiss_kmeans(X, n_clusters, n_threads=n_threads)
    # t2 = timeit.default_timer()
    # print('Faiss k-mean Time: {}'.format(t2 - t1))
    #
    # faiss_kmeans_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in faiss_kmeans_ans))

    """ scikit kmean++ """
    # t1 = timeit.default_timer()
    # kmeans = KMeans(init='random', n_clusters=n_clusters, n_init=4, max_iter=n_iter, random_state=0).fit(X)
    # t2 = timeit.default_timer()
    # print('scikit kmean++ Time: {}'.format(t2 - t1))
    #
    # sci_kmean_ans = getMetric(kmeans.labels_, true_labels)
    # print(' '.join(f"{x:.4f}" for x in sci_kmean_ans))

    """ scikit spectral clustering needs O(n^2) dense - so use sparse implemented version """
    ### This takes significant time, perhaps due to the cost of exact kNN graph
    # t1 = timeit.default_timer()
    # metric = "euclidean"
    # labels = utils.spectral_clustering(
    #     X,
    #     n_clusters=n_clusters,
    #     k=20,
    #     metric=metric,
    #     mutual=False,          # try False for symmetric-kNN
    #     sigma="auto",         # or a float, or ("median-k", 2.0)
    #     laplacian="sym",
    #     random_state=0
    # )
    # t2 = timeit.default_timer()
    # print('Sparse spectral clustering Time: {}'.format(t2 - t1))
    #
    # spectral_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in spectral_ans))
    #
    # # There is -1 as it is not connected to the largest component, need to increase k
    # vals, counts = np.unique(labels, return_counts=True)
    # for v, c in zip(vals, counts):
    #     print(f"{v}: {c}")

    """ Nystrom kernel kmean++ """
    # # Compute pairwise Euclidean distances over Subsample to avoid O(n^2) for large MNIST
    # n_samples = 100
    # X_sample = X[np.random.choice(len(X), n_samples, replace=False)]
    # dists = pairwise_distances(X_sample, metric="euclidean", n_jobs = n_threads)
    # median_dist = np.median(dists)
    # #
    # # Recommended gamma:
    # gamma = 1 / (2 * median_dist ** 2)
    #
    # print("Gamma: ", gamma)
    # print("n_samples: ", n_samples)
    #
    # t1 = timeit.default_timer()
    # labels, Z = utils.nystrom_kernel_kmeans(X, n_clusters=n_clusters, m=n_samples, gamma= gamma, n_iter=n_iter) # gamma = 1/ 2 sigma^2
    # t2 = timeit.default_timer()
    # print('Nystrom kernel k-mean Time: {}'.format(t2 - t1))
    #
    # nys_kmean_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in nys_kmean_ans))

    """ Nystrom spectral clustering """
    # Compute pairwise Euclidean distances over Subsample to avoid O(n^2) for large MNIST
    # n_samples = 1000
    # X_sample = X[np.random.choice(len(X), n_samples, replace=False)]
    # dists = pairwise_distances(X_sample, metric="euclidean", n_jobs = n_threads)
    # median_dist = np.median(dists)
    # #
    # # Recommended gamma:
    # gamma = 1 / (2 * median_dist ** 2)
    #
    # print("Gamma: ", gamma)
    # print("n_samples: ", n_samples)
    #
    # t1 = timeit.default_timer()
    # labels = utils.nystrom_spectral(X, k=n_clusters, m=n_samples, gamma= gamma, n_iter= n_iter)
    # t2 = timeit.default_timer()
    # print('Nystrom spectral k-mean Time: {}'.format(t2 - t1))
    #
    # nys_spectral_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in nys_spectral_ans))