import os

import utils
from utils import getMetric

import cluprop
import faiss
import networkx as nx

import numpy as np
import math
from sklearn.cluster import DBSCAN, OPTICS, KMeans, SpectralClustering,cluster_optics_dbscan
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.utils import shuffle
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from scipy.spatial.distance import jensenshannon

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.metrics.cluster import pair_confusion_matrix

from sklearn.neighbors import NearestNeighbors

from pynndescent import NNDescent

import timeit
import gc
from concurrent.futures import ThreadPoolExecutor


if __name__ == '__main__':

    path = "/work/Datasets/Clustering/"
    savePath = "/work/Datasets/Clustering/mnist8m_output/"
    bin_file = path + 'mnist8m_X.bin'

    n = 8100000
    d = 784

    true_labels = np.loadtxt(path + 'mnist8m_y_8100000_784')
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
    # n_iters = 2
    # max_cand = 100
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
    # np.save(savePath + f"nndescent_{n_trees}_{leafSize}_{n_iters}_{dist}_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath + f"nndescent_{n_trees}_{leafSize}_{n_iters}_{dist}_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32

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
    #     np.save(savePath + f"nndescent_{n_trees}_{leafSize}_{n_iters}_{dist}_{k_max}_{trial}_indices.npy", indices)    # shape: (n, k), dtype: int64
    #     np.save(savePath + f"nndescent_{n_trees}_{leafSize}_{n_iters}_{dist}_{k_max}_{trial}_distances.npy", distances)  # shape: (n, k), dtype: float32
    # exit()

    """ All ig.LPA with precomputed Faiss-IVFPQ/NNDescent/FalconnLite/CEOs """

    n_threads = 1
    k_max = 100
    dist = "cosine"

    # NNDescent params
    n_trees = 8
    leafSize = k_max # leafSize = int(k / n_trees)
    n_iters = 1
    # max_cand = 100


    # indices = np.load(savePath + "ivfpq_1024_10_8_Cosine_200_indices.npy")    # shape: (n, k), dtype: int64
    # distances = np.load(savePath + "ivfpq_1024_10_8_Cosine_200_distances.npy")  # shape: (n, k), dtype: float32

    print(f"RPT: metric={dist} n_trees={n_trees:2d} leafSize={leafSize:2d} n_iters={n_iters:2d} k_max={k_max:2d}")
    indices = np.load(savePath + f"nndescent_{n_trees}_{leafSize}_{n_iters}_{dist}_{k_max}_0_indices.npy")    # shape: (n, k), dtype: int32
    distances = np.load(savePath + f"nndescent_{n_trees}_{leafSize}_{n_iters}_{dist}_{k_max}_0_distances.npy")  # shape: (n, k), dtype: float32

    indices = indices.astype(np.int32)
    print(indices.shape)

    distances = distances.astype(np.float32)
    print(distances.shape)

    # k_max = 200
    # n_neighbors_list = [12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
    # n_neighbors_list = [10, 15, 20, 25, 30]
    # n_neighbors_list = [30, 40, 50]
    # n_neighbors_list = [60, 70, 80, 90, 100]
    n_neighbors_list = [50, 60, 70, 80]
    # n_neighbors_list = [50]
    # n_neighbors_list = [30]
    # n_neighbors_list = [x * 2 for x in n_neighbors_list]

    print(n_neighbors_list)

    dbs = cluprop.cluprop()
    dbs.min_cluster_size = 50
    dbs.n_threads = 16

    for n_neighbors in n_neighbors_list:

        print('n_neighbors: ', n_neighbors)
        K = min(n_neighbors + 1, k_max)

        # DANE
        t1 = timeit.default_timer()
        dbs.knn_dane(indices[:, 1 : K], distances[:, 1 : K], n_neighbors)
        t2 = timeit.default_timer()
        print('Dane Time: {}'.format(t2 - t1))
        lpa_ans = getMetric(np.array(dbs.labels_), true_labels)
        print(' '.join(f"{x:.4f}" for x in lpa_ans))


        # t1 = timeit.default_timer()
        # # weighted_graph = utils.igraph_form_weighted_sym_KNN_graph(indices[:, 1 : K], distances[:, 1 : K],verbose=True) # take lots of time
        # # weighted_graph = utils.igraph_form_weighted_sym_KNN_graph_fast(indices[:, 1 : K], distances[:, 1 : K],verbose=True) # also take lot of time
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
    # Compute pairwise Euclidean distances over Subsample to avoid O(n^2) for large MNIST
    # X_sample = X[np.random.choice(len(X), 1000, replace=False)]
    # dists = pairwise_distances(X_sample, metric="euclidean")
    # median_dist = np.median(dists)
    # #
    # # Recommended gamma:
    # gamma = 1 / (2 * median_dist ** 2)
    #
    # n_samples = round(0.0001 * n)
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
    # n_samples = round(0.001 * n)
    # n_clusters = 10
    # sigma = 2600  # mnist: sigma = 2600 for L2
    # n_iter = 20
    # gamma = 0.41199748967360983
    #
    # t1 = timeit.default_timer()
    # labels = nystrom_spectral(X, k=n_clusters, m=n_samples, gamma= gamma, n_iter= n_iter)
    # t2 = timeit.default_timer()
    # print('Nystrom spectral k-mean Time: {}'.format(t2 - t1))
    #
    # nys_spectral_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in nys_spectral_ans))