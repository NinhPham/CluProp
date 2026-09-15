import os

import utils
from utils import getMetric
import igraph as ig

import cluprop
import faiss

import numpy as np
import math
from sklearn.preprocessing import normalize

from sklearn.neighbors import NearestNeighbors
from pynndescent import NNDescent

import timeit
from pathlib import Path

if __name__ == '__main__':

    path = Path("~/Work/Datasets/Clustering/").expanduser()
    savePath = path / "mnist70K_output"

    # dataset = np.loadtxt(path + 'mnist_all_X')
    # X = np.loadtxt(path + 'mnist_all_X', delimiter=",")
    # X.dtype == np.float32
    # n, d = X.shape

    n = 70000
    d = 784

    bin_file = path / 'mnist_all_X.bin'
    X = utils.mmap_bin(path / 'mnist_all_X.bin', n, d)
    X.dtype == np.float32

    ### Preprocess data according to different metrics
    # ## Cosine
    # X = normalize(X, norm='l2', axis=1)

    ## For JS and Chi2
    # X = normalize(X, norm='l1', axis=1)
    # X /= X.sum(axis=1, keepdims=True) # Normalize each row to sum to 1 (L1 normalization)
    # nan_mask = np.isnan(X)
    # print(f"NaN mask: {nan_mask}")
    #
    # nan_indices = np.where(nan_mask)
    # print(f"Indices of NaN values: {nan_indices}")

    true_labels = np.loadtxt(path / 'mnist_all_y_70K_784', dtype=np.int32)

    n_clusters = 10
    n_iter = 20
    n_threads = 8
    n_repeats = 5

    """====================="""

    """ Compute exact kNN """
    # n_threads = 8
    # k_max = 20

    # # Exact Cosine
    # X = normalize(X, norm='l2', axis=1)
    # indices, distances = utils.faiss_kNN(X, k=k_max + 1, n_threads=n_threads)
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    # np.save(savePath / f"exact_cosine_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int32
    # np.save(savePath / f"exact_cosine_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32
    #
    # # Exact L2
    # indices, distances = utils.faiss_kNN(X, k=k_max + 1, n_threads=n_threads)
    # np.save(savePath / f"exact_L2_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath / f"exact_L2_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32
    #
    # # Exact L1
    # nbrs = NearestNeighbors(n_neighbors=k_max + 1, metric='manhattan',n_jobs=n_threads).fit(X)
    # distances, indices = nbrs.kneighbors(X)
    # np.save(savePath / f"exact_L1_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath / f"exact_L1_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32

    """ Compute Faiss approx kNN (IVF and IVFPQ)"""
    # n_threads = 32
    # k_max = 200
    #
    # # Faiss params
    # nlist = 100
    # nprobe = 10
    # m = 8
    # #
    # # L2
    # indices, distances = utils.faiss_approx_kNN_IVF(X, k=k_max + 1, n_list = nlist, n_probe = nprobe, n_threads=n_threads)
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    # np.save(savePath / f"ivf_{nlist}_{nprobe}_L2_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath / f"ivf_{nlist}_{nprobe}_L2_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32
    #
    # indices, distances = utils.faiss_approx_kNN_IVFPQ(X, k=k_max + 1, n_subquantizer=m, n_list = nlist, n_probe = nprobe, n_threads=n_threads)
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    # np.save(savePath / f"ivfpq_{nlist}_{nprobe}_{m}_L2_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath / f"ivfpq_{nlist}_{nprobe}_{m}_L2_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32
    #
    # # Cosine
    # X = normalize(X, norm='l2', axis=1)
    #
    # indices, distances = utils.faiss_approx_kNN_IVF(X, k=k_max + 1, n_list = nlist, n_probe = nprobe, n_threads=n_threads)
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    # np.save(savePath / f"ivf_{nlist}_{nprobe}_Cosine_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath / f"ivf_{nlist}_{nprobe}_Cosine_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32
    #
    # indices, distances = utils.faiss_approx_kNN_IVFPQ(X, k=k_max + 1, n_subquantizer=m, n_list = nlist, n_probe = nprobe, n_threads=n_threads)
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    # np.save(savePath / f"ivfpq_{nlist}_{nprobe}_{m}_Cosine_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath / f"ivfpq_{nlist}_{nprobe}_{m}_Cosine_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32

    """ Compute NNDescent """
    # n_threads = 8
    # k_max = 20
    # dist = "cosine"
    #
    # # NNDescent params
    # n_trees = 8
    # n_iters = 5
    # leafSize = 50
    # t1 = timeit.default_timer()
    #
    # indices, distances = NNDescent(X, n_neighbors=k_max, random_state=None,
    #                            n_trees=n_trees,          # <-- number of RP trees (you choose)
    #                            leaf_size=leafSize,        # good rule: ≈ n_neighbors
    #                            metric=dist, n_iters=n_iters, n_jobs=n_threads).neighbor_graph
    #
    # build_time = timeit.default_timer() - t1
    # # exact_kNN = np.load(savePath / "exact_Cosine_200_indices.npy").astype(np.int32)
    # # r = utils.getAcc_kNNG(exact_kNN[:,1:k_max+1], indices) # exact includes the index of the point itself
    # # print(f"RPT: n_trees={n_trees:2d} n_iters={n_iters:2d}  recall@{k_max}: {r:.4f} time={build_time:.4f}s")
    #
    # print(f"RPT: metric={dist} n_trees={n_trees:2d} n_iters={n_iters:2d} leafSize={leafSize:2d} time={build_time:.4f}s")
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)

    # np.save(savePath / f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int32
    # np.save(savePath / f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32

    """====================="""

    """ Propagation with precomputed EXACT/Faiss/NNDescent symmetric kNN (need +1 as Faiss/NNDescent consider the point itself as part of kNN) """
    n_threads = 8
    n_repeats = 1

    # Load precompute kNNG

    ## Exact params
    dist = "cosine"
    k_max = 200

    ## Faiss params
    # dist = "euclidean"
    # nlist = 100
    # nprobe = 10
    # k_max = 50

    ## NNDescent params
    # n_trees = 8
    # n_iters = 2
    # leafSize = 50
    # dist = "euclidean"
    # k_max = 20

    # indices = np.load(savePath / f"ivf_{nlist}_{nprobe}_{dist}_{k_max}_indices.npy")    # shape: (n, k), dtype: int64
    # distances = np.load(savePath / f"ivf_{nlist}_{nprobe}_{dist}_{k_max}_distances.npy")  # shape: (n, k), dtype: float32
    indices = np.load(savePath / f"exact_{dist}_{k_max}_indices.npy")    # shape: (n, k), dtype: int64
    distances = np.load(savePath / f"exact_{dist}_{k_max}_distances.npy")  # shape: (n, k), dtype: float32

    # indices = np.load(savePath / f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_indices.npy")    # shape: (n, k), dtype: int64
    # distances = np.load(savePath / f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_distances.npy")  # shape: (n, k), dtype: float32

    n_neighbors_list = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    # n_neighbors_list = [4, 6, 8, 10, 12, 14]
    # n_neighbors_list = [20, 25, 30, 35, 40, 45, 50]
    # n_neighbors_list = [60, 70, 80, 90, 100]
    # n_neighbors_list = [8]

    for n_neighbors in n_neighbors_list:

        print('n_neighbors: ', n_neighbors)
        K = min(n_neighbors + 1, k_max) # Faiss, NNDescent: + 1

        # unweighted_graph = utils.fast_unweighted_sym_knng_igraph(indices[:, 1 : K], verbose=False)
        #
        # for i in range(n_repeats):
        #
        #     t1 = timeit.default_timer()
        #     labels = utils.run_LPA(unweighted_graph)
        #     t2 = timeit.default_timer()
        #     print('LPA Time: {}'.format(t2 - t1))
        #     lpa_ans = getMetric(labels, true_labels)
        #     print(' '.join(f"{x:.4f}" for x in lpa_ans))

        # Note: exp_weight=False gives slightly higher accuracy
        # Leiden
        t1 = timeit.default_timer()
        weighted_graph = utils.fast_weighted_sym_knng_igraph(indices[:, 1 : K], distances[:, 1 : K], use_exp_weight=False,verbose=False)
        t2 = timeit.default_timer()
        print('Graph Construction Time: {}'.format(t2 - t1))

        # Mutual G_k
        # weighted_graph = utils.fast_weighted_mutual_knng_igraph(indices[:, 1 : K], distances[:, 1 : K], use_exp_weight=False,verbose=False)

        for i in range(n_repeats):

            t1 = timeit.default_timer()
            labels = utils.run_leiden(weighted_graph)
            t2 = timeit.default_timer()
            print('Leiden Time: {}'.format(t2 - t1))
            lpa_ans = getMetric(labels, true_labels)
            print(' '.join(f"{x:.4f}" for x in lpa_ans))

        # # Louvain
        # weighted_graph = utils.fast_weighted_sym_knng_igraph(indices[:, :n_neighbors], distances[:, :n_neighbors], use_exp_weight=False,verbose=False)
        # for i in range(n_repeats):
        #
        #     t1 = timeit.default_timer()
        #     labels = utils.run_louvain(weighted_graph)
        #     t2 = timeit.default_timer()
        #     print('Louvain Time: {}'.format(t2 - t1))
        #     lpa_ans = getMetric(labels, true_labels)
        #     print(' '.join(f"{x:.4f}" for x in lpa_ans))

        # DANE
        t1 = timeit.default_timer()
        dbs = cluprop.cluprop()
        dbs.knn_dane(indices[:, 1 : K], distances[:, 1 : K], K)
        t2 = timeit.default_timer()
        print('Dane Time: {}'.format(t2 - t1))
        lpa_ans = getMetric(np.array(dbs.labels_), true_labels)
        print(' '.join(f"{x:.4f}" for x in lpa_ans))

        # Remove noise
        dane_label = np.asarray(dbs.labels_).copy()
        n = len(dane_label)
        unique_labels, counts = np.unique(dane_label, return_counts=True)
        small_clusters = unique_labels[counts < 0.01 * n]
        small_clusters = small_clusters[small_clusters != -1]
        dane_label[np.isin(dane_label, small_clusters)] = -1

        # mask = dane_label != -1
        # dane_label = dane_label[mask]
        # y_new = true_labels[mask]
        # lpa_ans = getMetric(np.array(dane_label), y_new)

        noise_ratio = np.mean(dane_label == -1)
        # print("Number of noisy points:", num_noise)
        # print("Noise ratio:", noise_ratio)
        print(f"Noise percentage: {100 * noise_ratio:.2f}%")

        lpa_ans = getMetric(np.array(dane_label), true_labels)
        print(' '.join(f"{x:.4f}" for x in lpa_ans))

        # k_support
        # dbs.knn_dane(indices[:, : k_max + 1], distances[:, : k_max + 1], n_neighbors, round(1.5 * n_neighbors))
        # lpa_ans = getMetric(np.array(dbs.labels_), true_labels)
        # print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """====================="""

    """ DPC (Use n^2 memory) """
    # centers, labels = utils.density_peak_eps(X, percentile=2.0, top_k=n_clusters, plot_decision=False)

    """ Hdbscan (not support multi-threading) """
    # t1 = timeit.default_timer()
    # # Run HDBSCAN directly
    # clusterer = hdbscan.HDBSCAN(
    #     min_cluster_size=100,  # Minimum cluster size
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

    """ sOptics and sngOptics"""
    # eps = 0.25
    # dist = "Cosine"
    # sigma = 40 # not used
    # minPts = 50
    # n_threads = 8
    # k = 5
    # m = 50
    # utils.run_sOptics(X, minPts, eps, dist, k, m, sigma, n_threads)

    """ sDbscan"""
    # dist = "Cosine"
    # dist = "L2"
    # sigma = 40 # not used
    # k = 5
    # m = 50
    # n_threads = 8
    # for i in range(5):
    #     minPts_list = [12, 16, 20, 24, 28, 32]
    #     eps_list = [0.1, 0.11, 0.112, 0.13, 0.14, 0.15]
    #     for minPts in minPts_list:
    #         print("minPts: ", minPts)
    #         for eps in eps_list:
    #             labels = utils.run_sDbscan(X, minPts, eps, dist, k, m, sigma, n_threads)
    #             ans = getMetric(labels, true_labels)
    #             print(' '.join(f"{x:.4f}" for x in ans))

    #
    # dist = "JS"
    # for i in range(5):
    #     minPts_list = [12, 16, 20, 24, 28, 32]
    #     eps_list = [0.1, 0.11, 0.112, 0.13, 0.14, 0.15]
    #     for minPts in minPts_list:
    #         print("minPts: ", minPts)
    #         for eps in eps_list:
    #             labels = utils.run_sDbscan(X, minPts, eps, dist, k, m, sigma, n_threads)
    #             ans = getMetric(labels, true_labels)
    #             print(' '.join(f"{x:.4f}" for x in ans))
    #
    # dist = "L2"
    # sigma = 2600
    # for i in range(5):
    #     minPts_list = [12, 16, 20, 24, 28, 32]
    #     eps_list = [1150, 1200, 1250, 1300, 1350, 1400]
    #     for minPts in minPts_list:
    #         print("minPts: ", minPts)
    #         for eps in eps_list:
    #             labels = utils.run_sDbscan(X, minPts, eps, dist, k, m, sigma, n_threads)
    #             ans = getMetric(labels, true_labels)
    #             print(' '.join(f"{x:.4f}" for x in ans))

    # dist = "L1"
    # sigma = 30000
    # for i in range(5):
    #     minPts_list = [4, 6, 8, 10, 12, 14]
    #     eps_list = [5000, 6000, 7000, 8000, 9000, 10000]
    #     for minPts in minPts_list:
    #         print("minPts: ", minPts)
    #         for eps in eps_list:
    #             labels = utils.run_sDbscan(X, minPts, eps, dist, k, m, sigma, n_threads)
    #             ans = getMetric(labels, true_labels)
    #             print(' '.join(f"{x:.4f}" for x in ans))

    # utils.run_sDbscan(X, minPts=50, eps=0.13, dist="Cosine", k=50, m=50, sigma, n_threads = 32)

    """ sngDbscan"""
    # dist = "JS"
    # n_threads = 8
    # p = 0.01
    # for i in range(5):
    #     minPts_list = [12, 16, 20, 24, 28, 32]
    #     eps_list = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
    #     for minPts in minPts_list:
    #         print("minPts: ", minPts)
    #         for eps in eps_list:
    #             labels = utils.run_sngDbscan(X, minPts, eps, dist, p, n_threads)
    #             ans = getMetric(labels, true_labels)
    #             print(' '.join(f"{x:.4f}" for x in ans))

    # utils.run_sngDbscan(X, minPts=24, eps=0.13, dist="Cosine", p, n_threads)

    """ Dbscan via sngDbscan """
    # dist = "JS"
    # n_threads = 8
    # p = 1
    # for i in range(1):
    #     minPts_list = [12, 16, 20, 24, 28, 32]
    #     eps_list = [0.11]
    #     for minPts in minPts_list:
    #         print("minPts: ", minPts)
    #         for eps in eps_list:
    #             labels = utils.run_sngDbscan(X, minPts, eps, dist, p, n_threads)
    #             ans = getMetric(labels, true_labels)
    #             print(' '.join(f"{x:.4f}" for x in ans))

    """====================="""
