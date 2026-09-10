import os

import utils
from utils import getMetric
import igraph as ig

import cluprop

import numpy as np
import math

from sklearn.preprocessing import normalize
from pynndescent import NNDescent
import timeit

from pathlib import Path
from scipy.io import loadmat

if __name__ == '__main__':

    path = Path("~/Work/Datasets/Clustering/").expanduser()
    savePath = path / "covertype_output"

    dataName = "Covertype"
    mat = loadmat(path / f"data_{dataName}.mat")

    X = mat["fea"]
    true_labels = mat["gt"].ravel()

    # print("X shape:", X.shape)
    # print("y shape:", true_labels.shape)
    # print("first 10 labels:", true_labels[:10])

    n, d = X.shape

    n_clusters = 7
    n_threads = 8

    """====================="""
    """ Compute NNDescent """
    n_threads = 8
    k_max = 50
    # seed = 42
    n_trees = 8
    n_iters = 5
    dist = "euclidean"
    leafSize = 100
    max_cand = 100

    # If cosine, then call this function
    if (dist == "cosine"):
        X = normalize(X, norm='l2', axis=1)

    t1 = timeit.default_timer()

    # It does not count the point itself
    indices, distances = NNDescent(X, n_neighbors=k_max, random_state=None,
                               n_trees=n_trees,          # <-- number of RP trees (you choose)
                               leaf_size=leafSize,        # good rule: ≈ n_neighbors
                               max_candidates = max_cand, # "self-join" size of max 50 points
                               metric=dist, n_iters=n_iters, n_jobs=n_threads).neighbor_graph

    build_time = timeit.default_timer() - t1

    print(f"RPT: metric={dist} n_trees={n_trees:2d} n_iters={n_iters:2d} leafSize={leafSize:2d} kmax = {k_max:2d} time={build_time:.4f}s")
    indices = indices.astype(np.int32)
    distances = distances.astype(np.float32)

    np.save(savePath / f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int32
    np.save(savePath / f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32

    """====================="""

    """ iGraph propagation with precomputed NNDescent symmetric kNN (need +1 as Faiss/PyNNDescent consider the point itself as part of kNN) """
    n_threads = 8
    n_repeats = 1

    # NNDescent params
    n_trees = 8
    n_iters = 1
    leafSize = 100
    dist = "euclidean"
    k_max = 50

    # Load precompute kNN from PyNNDescent
    indices = np.load(savePath / f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_indices.npy")    # shape: (n, k), dtype: int64
    distances = np.load(savePath / f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_distances.npy")  # shape: (n, k), dtype: float32

    # n_neighbors_list = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    n_neighbors_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for n_neighbors in n_neighbors_list:

        print('n_neighbors: ', n_neighbors)
        K = min(n_neighbors + 1, k_max) # Faiss consider the point itself as NN

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
            labels = utils.run_leiden(weighted_graph)
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


    """====================="""

    """ DANE with precomputed kNN by PyNNDescent """
    # n_threads = 8
    # k_max = 20
    #
    # # indices = np.load(savePath / f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_indices.npy")    # shape: (n, k), dtype: int64
    # # distances = np.load(savePath / f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_distances.npy")  # shape: (n, k), dtype: float32
    #
    n_neighbors_list = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    # # # n_neighbors_list = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # # # n_neighbors_list = [4, 5, 6, 7, 8, 9]
    # n_neighbors_list = [1, 2, 3, 4, 5, 6, 7, 8]
    #

    dbs = cluprop.cluprop()
    dbs.n_threads = 8

    for n_neighbors in n_neighbors_list:

        print('n_neighbors: ', n_neighbors) # k' in the paper

        K = min(n_neighbors + 1, k_max)

        # cluprop
        t1 = timeit.default_timer()

        # G_K where K = ck
        dbs.knn_dane(indices[:, 1 : K], distances[:, 1 : K], n_neighbors)
        t2 = timeit.default_timer()
        print('Dane Time: {}'.format(t2 - t1))
        dane_ans = getMetric(np.array(dbs.labels_), true_labels)
        print(' '.join(f"{x:.4f}" for x in dane_ans))
