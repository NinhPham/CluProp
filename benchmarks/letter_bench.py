import os

import utils
from utils import getMetric
import faiss
import numpy as np
import math
from sklearn.preprocessing import normalize
from pynndescent import NNDescent

import timeit
from pathlib import Path

if __name__ == '__main__':

    savePath = Path("~/Work/Datasets/Clustering/letter_output").expanduser()

    X = np.loadtxt("test/Dataset/letter-data.txt", delimiter=",")
    true_labels = np.loadtxt("test/Dataset/letter-labels.txt", delimiter=",")

    print("X shape:", X.shape)
    print("y shape:", true_labels.shape)

    n_clusters = len(np.unique(true_labels))
    print("Number of clusters: ", n_clusters)

    n, d = X.shape


    """====================="""
    """ Compute exact kNN """
    n_threads = 8
    k_max = 20

    # Exact L2
    # indices, distances = utils.faiss_kNN(X, k=k_max + 1, n_threads=n_threads)
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    # np.save(savePath / f"exact_euclidean_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath / f"exact_euclidean_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32

    X = normalize(X, norm='l2', axis=1)
    indices, distances = utils.faiss_kNN(X, k=k_max + 1, n_threads=n_threads)

    indices = indices.astype(np.int32)
    distances = distances.astype(np.float32)
    np.save(savePath / f"exact_cosine_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int32
    np.save(savePath / f"exact_cosine_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32

    """====================="""
    n_clusters = 26

    n_iter = 20
    n_threads = 8
    n_repeats = 5

    """====================="""
    """ Compute NNDescent """
    # n_threads = 8
    # k_max = 20
    # seed = 42
    #
    # # NNDescent params
    # # X = normalize(X, norm='l2', axis=1)
    #
    # n_trees = 8
    # n_iters = 5
    # dist = "euclidean"
    # leafSize = 100
    # # max_cand = 100
    #
    # t1 = timeit.default_timer()
    #
    # # It does not count the point itself
    # # indices, distances = NNDescent(X, n_neighbors=k_max, random_state=None,
    # #                            n_trees=n_trees,          # <-- number of RP trees (you choose)
    # #                            leaf_size=leafSize,        # good rule: ≈ n_neighbors
    # #                            # max_candidates = max_cand, # "self-join" size of max 50 points
    # #                            metric=dist, n_iters=n_iters, n_jobs=n_threads).neighbor_graph
    #
    # build_time = timeit.default_timer() - t1
    # # exact_kNN = np.load(savePath / "exact_Cosine_200_indices.npy").astype(np.int32)
    # # r = utils.getAcc_kNNG(exact_kNN[:,1:k_max+1], indices) # exact includes the index of the point itself
    # # print(f"RPT: n_trees={n_trees:2d} n_iters={n_iters:2d}  recall@{k_max}: {r:.4f} time={build_time:.4f}s")
    #
    # print(f"RPT: metric={dist} n_trees={n_trees:2d} n_iters={n_iters:2d} leafSize={leafSize:2d} time={build_time:.4f}s")

    #
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    #
    # np.save(savePath / f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int32
    # np.save(savePath / f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32

    """====================="""

    """ iGraph propagation with precomputed EXACT/Faiss/NNDescent symmetric kNN (need +1 as Faiss consider the point itself as part of kNN) """
    n_threads = 8
    n_repeats = 1
    dist = "cosine"
    k_max = 20

    # NNDescent params
    # n_trees = 8
    # n_iters = 5
    # leafSize = 100


    # Load precompute kNNG
    indices = np.load(savePath / f"exact_{dist}_{k_max}_indices.npy")    # shape: (n, k), dtype: int64
    distances = np.load(savePath / f"exact_{dist}_{k_max}_distances.npy")  # shape: (n, k), dtype: float32

    # indices = np.load(savePath / f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_indices.npy")    # shape: (n, k), dtype: int64
    # distances = np.load(savePath / f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_distances.npy")  # shape: (n, k), dtype: float32


    n_neighbors_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for n_neighbors in n_neighbors_list:

        print('n_neighbors: ', n_neighbors)
        K = min(n_neighbors + 1, k_max)

        # LPA: need + 1 for Faiss
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

        # Note: exp_weight=False gives slightly higher accuracy, need + 1 for Faiss
        # Leiden
        # This is G_k
        t1 = timeit.default_timer()
        weighted_graph = utils.fast_weighted_sym_knng_igraph(indices[:, 1 : K], distances[:, 1 : K], use_exp_weight=False,verbose=False)
        t2 = timeit.default_timer()
        print('Graph Construction Time: {}'.format(t2 - t1))

        # Mutual G_k
        # weighted_graph = utils.fast_weighted_mutual_knng_igraph(indices[:, 1 : K], distances[:, 1 : K], use_exp_weight=False,verbose=False)

        for i in range(n_repeats):

            t1 = timeit.default_timer()
            labels = utils.run_leiden(weighted_graph, resolution=1.0)
            t2 = timeit.default_timer()
            print('Leiden Time: {}'.format(t2 - t1))
            lpa_ans = getMetric(labels, true_labels)
            print(' '.join(f"{x:.4f}" for x in lpa_ans))

        # Louvain
        # This is G_k
        # weighted_graph = utils.fast_weighted_sym_knng_igraph(indices[:, :n_neighbors], distances[:, :n_neighbors], use_exp_weight=False,verbose=False)

        # for i in range(n_repeats):
        #
        #     t1 = timeit.default_timer()
        #     labels = utils.run_louvain(weighted_graph)
        #     t2 = timeit.default_timer()
        #     print('Louvain Time: {}'.format(t2 - t1))
        #     lpa_ans = getMetric(labels, true_labels)
        #     print(' '.join(f"{x:.4f}" for x in lpa_ans))

