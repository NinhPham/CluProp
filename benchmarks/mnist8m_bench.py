import os

import utils
from utils import getMetric

import cluprop
import faiss

import numpy as np
import math
from sklearn.cluster import KMeans

from sklearn.preprocessing import normalize
from pynndescent import NNDescent

import timeit

from pathlib import Path
from scipy.io import loadmat

if __name__ == '__main__':

    path = Path("~/Work/Datasets/Clustering/").expanduser()
    savePath = path / "mnist8m_output"


    bin_file = path / 'mnist8m_X.bin'
    n = 8100000
    d = 784

    true_labels = np.loadtxt(path / 'mnist8m_y_8100000_784')
    n_clusters = 10
    n_iter = 20

    """ Compute NNDescent """
    # n_repeats = 5
    # n_threads = 128
    # k_max = 100
    # seed = 42
    # dist = "cosine"
    #
    # # NNDescent params
    # n_trees = 8
    # leafSize = k_max # leafSize = int(k / n_trees)
    # n_iters = 1
    # max_cand = k_max
    #
    # X = utils.mmap_bin(bin_file, n, d)
    # X = X.astype(np.float32)
    # print("Finish reading data")
    #
    # t1 = timeit.default_timer()
    #
    # indices, distances = NNDescent(X, n_neighbors=k_max, random_state=seed,
    #                                n_trees=n_trees,          # <-- number of RP trees (you choose)
    #                                leaf_size=leafSize,        # good rule: ≈ n_neighbors
    #                                max_candidates = max_cand, # "self-join" size of max 50 points
    #                                metric=dist, n_iters=n_iters, n_jobs=n_threads).neighbor_graph
    # del X
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    # np.save(savePath / f"nndescent_{n_trees}_{leafSize}_{n_iters}_{dist}_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath / f"nndescent_{n_trees}_{leafSize}_{n_iters}_{dist}_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32

    # for trial in range(n_repeats):
    #     indices, distances = NNDescent(X, n_neighbors=k_max,
    #                                    n_trees=n_trees,          # <-- number of RP trees (you choose)
    #                                    leaf_size=leafSize,        # good rule: ≈ n_neighbors
    #                                    max_candidates = max_cand, # "self-join" size of max 50 points
    #                                    metric=dist, n_iters=n_iters, n_jobs=n_threads).neighbor_graph
    #
    #     build_time = timeit.default_timer() - t1
    #     print(f"RPT: metric={dist} n_trees={n_trees:2d} n_iters={n_iters:2d} k_max={k_max:2d} max_cand={max_cand:2d} time={build_time:.4f}s")
    #
    #     indices = indices.astype(np.int32)
    #     distances = distances.astype(np.float32)
    #
    #     np.save(savePath / f"nndescent_{n_trees}_{leafSize}_{n_iters}_{dist}_{k_max}_{trial}_indices.npy", indices)    # shape: (n, k), dtype: int64
    #     np.save(savePath / f"nndescent_{n_trees}_{leafSize}_{n_iters}_{dist}_{k_max}_{trial}_distances.npy", distances)  # shape: (n, k), dtype: float32
    # exit()

    """ All ig.LPA with precomputed NNDescent """

    n_threads = 8
    k_max = 80
    dist = "cosine"

    # NNDescent params
    n_trees = 8
    leafSize = k_max
    n_iters = 1

    print(f"RPT: metric={dist} n_trees={n_trees:2d} leafSize={leafSize:2d} n_iters={n_iters:2d} k_max={k_max:2d}")
    indices = np.load(savePath / f"nndescent_{n_trees}_{leafSize}_{n_iters}_{dist}_{k_max}_indices.npy")    # shape: (n, k), dtype: int32
    distances = np.load(savePath / f"nndescent_{n_trees}_{leafSize}_{n_iters}_{dist}_{k_max}_distances.npy")  # shape: (n, k), dtype: float32

    indices = indices.astype(np.int32)
    print(indices.shape)
    distances = distances.astype(np.float32)
    print(distances.shape)

    # n_neighbors_list = [10, 20, 30, 40, 50]
    n_neighbors_list = [50, 60, 70, 80]
    print(n_neighbors_list)

    dbs = cluprop.cluprop()
    # dbs.min_cluster_size = 50
    dbs.n_threads = 8

    for n_neighbors in n_neighbors_list:

        print('n_neighbors: ', n_neighbors)
        K = min(n_neighbors + 1, k_max) # PyNNDescent and Faiss consider the point itself in kNN

        """ Dane """
        t1 = timeit.default_timer()
        dbs.knn_dane(indices[:, 1 : K], distances[:, 1 : K], n_neighbors)
        t2 = timeit.default_timer()
        print('Dane Time: {}'.format(t2 - t1))
        lpa_ans = getMetric(np.array(dbs.labels_), true_labels)
        print(' '.join(f"{x:.4f}" for x in lpa_ans))

        # Remove noise
        dane_label = np.asarray(dbs.labels_).copy()
        n = len(dane_label)
        unique_labels, counts = np.unique(dane_label, return_counts=True)
        small_clusters = unique_labels[counts < 0.001 * n]
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

        """ Leiden """
        # t1 = timeit.default_timer()
        # weighted_graph = utils.fast_weighted_sym_knng_igraph(indices[:, 1:K], distances[:, 1:K], use_exp_weight=False, verbose = True) # fastest - sometime not work for k = 50
        #
        # t2 = timeit.default_timer()
        # print('Graph Construction Time: {}'.format(t2 - t1))
        #
        # # del indices, distances
        #
        # #### Leiden
        # t1 = timeit.default_timer()
        # labels = utils.run_leiden(weighted_graph)
        # t2 = timeit.default_timer()
        # print('Leiden Time: {}'.format(t2 - t1))
        # lpa_ans = getMetric(labels, true_labels)
        # print(' '.join(f"{x:.4f}" for x in lpa_ans))

        # Louvain
        # t1 = timeit.default_timer()
        # labels = utils.run_louvain(weighted_graph)
        # t2 = timeit.default_timer()
        # print('Louvain Time: {}'.format(t2 - t1))
        # lpa_ans = getMetric(labels, true_labels)
        # print(' '.join(f"{x:.4f}" for x in lpa_ans))


    """ faiss k-mean """
    # t1 = timeit.default_timer()
    # labels = utils.faiss_kmeans(X, n_clusters)
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



