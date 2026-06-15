import os

import utils
from utils import getMetric
import networkx as nx
import igraph as ig

import clupig
import faiss

import numpy as np
import math
from sklearn.cluster import DBSCAN, OPTICS, KMeans, SpectralClustering, cluster_optics_dbscan
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

def js_distance(x, y):
    return jensenshannon(x, y, base=2.0)  # base 2, returns sqrt(JS divergence)

if __name__ == '__main__':

    path = "//home/npha145/Uni of Auckland Dropbox/Ninh Pham/Working/_Code/Matlab/USPEC/"
    savePath = "/shared/Dataset/Clustering/usps_output/"

    from scipy.io import loadmat

    dataName = "USPS"
    mat = loadmat(f"{path}data_{dataName}.mat")

    X = mat["fea"]
    true_labels = mat["gt"].ravel()

    print("X shape:", X.shape)
    print("y shape:", true_labels.shape)
    print("first 10 labels:", true_labels[:10])

    n, d = X.shape


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


    n_clusters = 10

    n_iter = 20
    n_threads = 8
    n_repeats = 5

    """====================="""
    """ Compute exact kNN """
    n_threads = 8
    k_max = 20

    # Exact L2
    # indices, distances = utils.faiss_kNN(X, k=k_max + 1, n_threads=n_threads)
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    # np.save(savePath + f"exact_euclidean_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath + f"exact_euclidean_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32

    # X = normalize(X, norm='l2', axis=1)
    # indices, distances = utils.faiss_kNN(X, k=k_max + 1, n_threads=n_threads)
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    # np.save(savePath + f"exact_cosine_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int32
    # np.save(savePath + f"exact_cosine_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32

    """====================="""
    """ Compute NNDescent """
    # n_threads = 8
    # k_max = 50
    # seed = 42
    #
    # # NNDescent params
    # # X = normalize(X, norm='l2', axis=1)
    #
    # n_trees = 8
    # n_iters = 5
    # dist = "cosine"
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
    # # exact_kNN = np.load(savePath + "exact_Cosine_200_indices.npy").astype(np.int32)
    # # r = utils.getAcc_kNNG(exact_kNN[:,1:k_max+1], indices) # exact includes the index of the point itself
    # # print(f"RPT: n_trees={n_trees:2d} n_iters={n_iters:2d}  recall@{k_max}: {r:.4f} time={build_time:.4f}s")
    #
    # print(f"RPT: metric={dist} n_trees={n_trees:2d} n_iters={n_iters:2d} leafSize={leafSize:2d} time={build_time:.4f}s")


    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)

    # np.save(savePath + f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int32
    # np.save(savePath + f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32

    """====================="""

    """ iGraph propagation with precomputed EXACT/Faiss/NNDescent symmetric kNN (need +1 as Faiss consider the point itself as part of kNN) """
    n_threads = 8
    n_repeats = 1
    k_max = 20
    dist = "cosine"

    # NNDescent params
    # n_trees = 8
    # n_iters = 5
    # leafSize = 100
    #
    #

    # Load precompute kNNG

    # indices = np.load(savePath + f"ivf_{nlist}_{nprobe}_{dist}_{k_max}_indices.npy")    # shape: (n, k), dtype: int64
    # distances = np.load(savePath + f"ivf_{nlist}_{nprobe}_{dist}_{k_max}_distances.npy")  # shape: (n, k), dtype: float32
    # indices = np.load(savePath + f"exact_{dist}_{k_max}_indices.npy")    # shape: (n, k), dtype: int64
    # distances = np.load(savePath + f"exact_{dist}_{k_max}_distances.npy")  # shape: (n, k), dtype: float32
    indices = np.load(savePath + f"exact_{dist}_{k_max}_indices.npy")    # shape: (n, k), dtype: int64
    distances = np.load(savePath + f"exact_{dist}_{k_max}_distances.npy")  # shape: (n, k), dtype: float32

    # indices = np.load(savePath + f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_indices.npy")    # shape: (n, k), dtype: int64
    # distances = np.load(savePath + f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_distances.npy")  # shape: (n, k), dtype: float32

    n_neighbors_list = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    # n_neighbors_list = [4, 6, 8, 10, 12, 14]
    # n_neighbors_list = [20, 25, 30, 35, 40, 45, 50]
    # n_neighbors_list = [60, 70, 80, 90, 100]
    # n_neighbors_list = [8]

    for n_neighbors in n_neighbors_list:

        print('n_neighbors: ', n_neighbors)
        K = min(n_neighbors + 1, k_max)  # +1 for faiss

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
        # weighted_graph = utils.fast_local_scaled_sym_knng_igraph(indices[:, 1: K], distances[:, 1: K], verbose=False)
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

    """ (c,k)-DNP with precomputed EXACT/Faiss symmetric kNN, needs +1 """
    """ c > 1 gives higher accuracy, and G_kmax where kmax > c*k gives more stable accuracy than G_k """
    # # n_threads = 8
    # # k_max = 200
    # # nlist = 100
    # # nprobe = 10
    # #
    # # # Load precompute kNNG
    # savePath = "/shared/Dataset/Clustering/mnist70K_output/"
    #
    # # indices = np.load(savePath + f"exact_Cosine_{k_max}_indices.npy")    # shape: (n, k), dtype: int64
    # # distances = np.load(savePath + f"exact_Cosine_{k_max}_distances.npy")  # shape: (n, k), dtype: float32
    # # # indices = np.load(savePath + f"ivf_{nlist}_{nprobe}_Cosine_{k_max}_indices.npy")    # shape: (n, k), dtype: int64
    # # # distances = np.load(savePath + f"ivf_{nlist}_{nprobe}_Cosine_{k_max}_distances.npy")  # shape: (n, k), dtype: float32
    #
    # indices = np.load(savePath + f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_indices.npy")    # shape: (n, k), dtype: int64
    # distances = np.load(savePath + f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_distances.npy")  # shape: (n, k), dtype: float32
    #
    # # # n_neighbors_list = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    # # # n_neighbors_list = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # # # n_neighbors_list = [4, 5, 6, 7, 8, 9]
    # # n_neighbors_list = [14]
    # # n_neighbors_list = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
    #
    # n_neighbors_list = [4, 6, 8, 10, 12, 14]
    # #
    #
    # c = 1
    # dbs = clupig.clupig(n, d)
    # # dbs.set_min_cluster_size(50)
    #
    # for n_neighbors in n_neighbors_list:
    #
    #     print('n_neighbors: ', n_neighbors) # k' in the paper
    #
    #     K = min(c * n_neighbors + 1, k_max)
    #
    #     # clupig
    #     t1 = timeit.default_timer()
    #
    #     # G_K where K = ck
    #     dbs.dnp_from_knn(indices[:, 1 : K], distances[:, 1 : K], n_neighbors, c=c)
    #     lpa_ans = getMetric(np.array(dbs.labels_), true_labels)
    #     print(' '.join(f"{x:.4f}" for x in lpa_ans))
    #
    #     # # G_kmax where ck <= K_max,
    #     # dbs.dnp_from_knn(indices[:, : k_max + 1], distances[:, : k_max + 1], n_neighbors, c=c)
    #     # lpa_ans = getMetric(np.array(dbs.labels_), true_labels)
    #     # print(' '.join(f"{x:.4f}" for x in lpa_ans))
    #
    #     t2 = timeit.default_timer()
    #     print('sVDC Time: {}'.format(t2 - t1))


    """====================="""