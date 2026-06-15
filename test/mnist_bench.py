import os

import utils
from utils import getMetric
import networkx as nx
import igraph as ig

import clupig
import faiss
import cluprop

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

    path = "/shared/Dataset/Clustering/"
    savePath = "/shared/Dataset/Clustering/mnist70K_output/"

    # dataset = np.loadtxt(path + 'mnist_all_X')
    # X = np.loadtxt(path + 'mnist_all_X', delimiter=",")
    # X.dtype == np.float32
    # n, d = X.shape

    n = 70000
    d = 784

    bin_file = path + 'mnist_all_X.bin'
    X = utils.mmap_bin(path + 'mnist_all_X.bin', n, d)
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

    true_labels = np.loadtxt(path + 'mnist_all_y_70K_784', dtype=np.int32)

    n_clusters = 10
    n_iter = 20
    n_threads = 8
    n_repeats = 5

    """====================="""

    """ Compute exact kNN """
    # n_threads = 8
    # k_max = 20


    # Exact L2
    # indices, distances = utils.faiss_kNN(X, k=k_max + 1, n_threads=n_threads)

    # np.save(savePath + f"exact_L2_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath + f"exact_L2_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32
    #
    # # Exact L1
    # nbrs = NearestNeighbors(n_neighbors=k_max + 1, metric='manhattan',n_jobs=n_threads).fit(X)
    # distances, indices = nbrs.kneighbors(X)
    # np.save(savePath + f"exact_L1_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath + f"exact_L1_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32
    #
    # # Exact JS
    # numProj = 128
    # top_s = 1
    # top_m = 1
    # top_p = 1
    #
    # ker_n_features = 1
    # sigma = 1 # only used on L1: 30000, L2: 2600
    # dist = "JS"
    # clusterNoise = 0 # not used on sOptics
    # output = 'clupig'
    # numThreads = 32
    # verbose = False
    # ker_intervalSampling = 0.4 # only used on Chi2, JS distances
    #
    # seed = -1  # -1 is random
    # dbs = clupig.clupig(n, d)
    # dbs.set_params(numProj, top_s, top_m, top_p, dist, ker_n_features, sigma, ker_intervalSampling, verbose, numThreads, seed, output)
    #
    # t1 = timeit.default_timer()
    # indices, distances = dbs.bf_kNN_from_file(bin_file, k_max)
    # t2 = timeit.default_timer()
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    # np.save(savePath + f"exact_JS_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath + f"exact_JS_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32

    # # Exact Cosine
    # X = normalize(X, norm='l2', axis=1)
    # indices, distances = utils.faiss_kNN(X, k=k_max + 1, n_threads=n_threads)
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    # np.save(savePath + f"exact_cosine_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int32
    # np.save(savePath + f"exact_cosine_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32

    """ Compute Faiss approx kNN (IVF and IVFPQ)"""
    # n_threads = 32
    # k_max = 200
    # savePath = "/shared/Dataset/Clustering/mnist70K_output/"
    #
    # # Faiss params
    # nlist = 100
    # nprobe = 10
    # m = 8
    # #
    # # # L2
    # indices, distances = utils.faiss_approx_kNN_IVF(X, k=k_max + 1, n_list = nlist, n_probe = nprobe, n_threads=n_threads)
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    # np.save(savePath + f"ivf_{nlist}_{nprobe}_L2_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath + f"ivf_{nlist}_{nprobe}_L2_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32
    #
    # indices, distances = utils.faiss_approx_kNN_IVFPQ(X, k=k_max + 1, n_subquantizer=m, n_list = nlist, n_probe = nprobe, n_threads=n_threads)
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    # np.save(savePath + f"ivfpq_{nlist}_{nprobe}_{m}_L2_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath + f"ivfpq_{nlist}_{nprobe}_{m}_L2_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32
    #
    # # Cosine
    # X = normalize(X, norm='l2', axis=1)
    #
    # indices, distances = utils.faiss_approx_kNN_IVF(X, k=k_max + 1, n_list = nlist, n_probe = nprobe, n_threads=n_threads)
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    # np.save(savePath + f"ivf_{nlist}_{nprobe}_Cosine_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath + f"ivf_{nlist}_{nprobe}_Cosine_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32
    #
    # indices, distances = utils.faiss_approx_kNN_IVFPQ(X, k=k_max + 1, n_subquantizer=m, n_list = nlist, n_probe = nprobe, n_threads=n_threads)
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    # np.save(savePath + f"ivfpq_{nlist}_{nprobe}_{m}_Cosine_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath + f"ivfpq_{nlist}_{nprobe}_{m}_Cosine_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32

    """ Compute NNDescent """
    n_threads = 8
    k_max = 20
    savePath = "/shared/Dataset/Clustering/mnist70K_output/"
    seed = 42
    dist = "cosine"

    # NNDescent params
    # X = normalize(X, norm='l2', axis=1)
    n_trees = 8
    n_iters = 5
    leafSize = 50
    t1 = timeit.default_timer()

    # It does not count the point itself
    indices, distances = NNDescent(X, n_neighbors=k_max, random_state=None,
                               n_trees=n_trees,          # <-- number of RP trees (you choose)
                               leaf_size=leafSize,        # good rule: ≈ n_neighbors
                               metric=dist, n_iters=n_iters, n_jobs=n_threads).neighbor_graph

    build_time = timeit.default_timer() - t1
    # exact_kNN = np.load(savePath + "exact_Cosine_200_indices.npy").astype(np.int32)
    # r = utils.getAcc_kNNG(exact_kNN[:,1:k_max+1], indices) # exact includes the index of the point itself
    # print(f"RPT: n_trees={n_trees:2d} n_iters={n_iters:2d}  recall@{k_max}: {r:.4f} time={build_time:.4f}s")

    print(f"RPT: metric={dist} n_trees={n_trees:2d} n_iters={n_iters:2d} leafSize={leafSize:2d} time={build_time:.4f}s")
    indices = indices.astype(np.int32)
    distances = distances.astype(np.float32)

    # np.save(savePath + f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int32
    # np.save(savePath + f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32

    """====================="""

    """ Propagation with precomputed EXACT/Faiss/NNDescent symmetric kNN (need +1 as Faiss consider the point itself as part of kNN) """
    n_threads = 8
    n_repeats = 1

    # Load precompute kNNG

    # Faiss params
    # dist = "BrayCurtis"
    # nlist = 100
    # nprobe = 10
    # k_max = 50

    # NNDescent params
    # n_trees = 8
    # n_iters = 2
    # leafSize = 50
    # dist = "euclidean"
    # k_max = 20

    # indices = np.load(savePath + f"ivf_{nlist}_{nprobe}_{dist}_{k_max}_indices.npy")    # shape: (n, k), dtype: int64
    # distances = np.load(savePath + f"ivf_{nlist}_{nprobe}_{dist}_{k_max}_distances.npy")  # shape: (n, k), dtype: float32
    # indices = np.load(savePath + f"exact_{dist}_{k_max}_indices.npy")    # shape: (n, k), dtype: int64
    # distances = np.load(savePath + f"exact_{dist}_{k_max}_distances.npy")  # shape: (n, k), dtype: float32

    # indices = np.load(savePath + f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_indices.npy")    # shape: (n, k), dtype: int64
    # distances = np.load(savePath + f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_distances.npy")  # shape: (n, k), dtype: float32

    # n_neighbors_list = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    n_neighbors_list = [4, 6, 8, 10, 12, 14]
    # n_neighbors_list = [20, 25, 30, 35, 40, 45, 50]
    # n_neighbors_list = [60, 70, 80, 90, 100]
    # n_neighbors_list = [8]

    for n_neighbors in n_neighbors_list:

        print('n_neighbors: ', n_neighbors)
        K = min(n_neighbors + 1, k_max) # Faiss, NNDescent: + 1
        k_expand = round(K / 1)

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

        for i in range(n_repeats):

            t1 = timeit.default_timer()
            labels = utils.run_louvain(weighted_graph)
            t2 = timeit.default_timer()
            print('Louvain Time: {}'.format(t2 - t1))
            lpa_ans = getMetric(labels, true_labels)
            print(' '.join(f"{x:.4f}" for x in lpa_ans))

        # DANE
        t1 = timeit.default_timer()

        dbs = cluprop.cluprop(n, d)
        dbs.knn_dane(indices[:, 1 : K], distances[:, 1 : K], n_neighbors)
        lpa_ans = getMetric(np.array(dbs.labels_), true_labels)
        print(' '.join(f"{x:.4f}" for x in lpa_ans))

        # # G_kmax where ck <= K_max,
        # dbs.dnp_from_knn(indices[:, : k_max + 1], distances[:, : k_max + 1], n_neighbors, c=c)
        # lpa_ans = getMetric(np.array(dbs.labels_), true_labels)
        # print(' '.join(f"{x:.4f}" for x in lpa_ans))

        t2 = timeit.default_timer()
        print('DANE Time: {}'.format(t2 - t1))

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

    """ (c,k)-DNP with CEOs (Cosine)"""
    """ Note: clupig forms cluster need c = 2 to have higher accuracy, so need to use 2 * n_neighbors when forming the graph """
    # n_threads = 32
    #
    # # Cosine or L2
    # numProj = 256
    # s = 20
    # m = 50
    # topP = 5
    #
    # numEmbed = 1024
    # sigma = 30000  # only used on L1: 30000, L2: 2600
    # dist = "Cosine"
    # output = 'clupig'
    # numThreads = n_threads
    # verbose = False
    # intervalSampling = 0.4  # only used on Chi2, JS distances
    #
    # seed = -1  # -1 is random
    # dbs = clupig.clupig(n, d)
    #
    # dbs.set_params(numProj, s, m, topP, dist, numEmbed, sigma, intervalSampling, verbose, numThreads, seed, output)
    # # dbs.set_min_cluster_size(10)
    # # dbs.set_neighbor_cutoff()
    #
    # # n_neighbors_list = [4, 5, 6, 7, 8, 9]
    # # n_neighbors_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    # # n_neighbors_list = [14, 16, 18, 20, 22, 24]
    # n_neighbors_list = [20]
    #
    # K = 40
    # c = 1
    #
    # print(n_neighbors_list)
    # print("c: ", c)
    #
    # n_repeats = 5
    #
    # for n_neighbors in n_neighbors_list:
    #
    #     # print('n_neighbors: ', n_neighbors)
    #
    #     for i in range(n_repeats):
    #
    #         t1 = timeit.default_timer()
    #
    #         # indices, distances = dbs.ceos2_knn_from_file(bin_file, K)
    #         # dbs.dnp_from_knn(indices, distances, n_neighbors, c=c)
    #
    #         dbs.ceos2_dnp(X, n_neighbors, c)
    #         # dbs.ceos2_dnp_from_file(bin_file, n_neighbors, c)
    #
    #         t2 = timeit.default_timer()
    #         # print('clupig Time: {}'.format(t2 - t1))
    #         lpa_ans = getMetric(np.array(dbs.labels_), true_labels)
    #         print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """====================="""

    """ DPC (Use n^2 memory) """
    # centers, labels = utils.density_peak_eps(X, percentile=2.0, top_k=n_clusters, plot_decision=False)

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
    # for i in range(5):
    #     minPts_list = [4, 6, 8, 10, 12, 14]
    #     eps_list = [5000, 6000, 7000, 8000, 9000, 10000]
    #     for minPts in minPts_list:
    #         print("minPts: ", minPts)
    #         for eps in eps_list:
    #             ans = run_sDbscan(X, minPts, eps, dist, sigma=30000, n_threads=8)
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

    """====================="""

    """ kNN nx.LPA <=> sys kNNG """
    # n_neighbors = 24
    # n_threads = 32
    # n_iter = 100
    # print("Neighbors: ", n_neighbors)
    #
    # t1 = timeit.default_timer()
    # G = utils.nx_form_unweighted_KNN_graph(X, k=n_neighbors, n_threads=n_threads)
    # t2 = timeit.default_timer()
    # print('Faiss Time: {}'.format(t2 - t1))
    # labels = utils.nx_LPA(G, max_iter = n_iter)
    # t2 = timeit.default_timer()
    # print('kNN LPA Time: {}'.format(t2 - t1))
    #
    # lpa_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """ kNN nx.LPA - REPEAT <=> sys kNNG - REPEAT """
    # k_max = 32
    # n_threads = 32
    # indices, distances = utils.faiss_kNN(X, k=k_max + 1, n_threads=n_threads)
    # n_neighbors_list = [12, 16, 20, 24, 28, 32]
    #
    # for n_neighbors in n_neighbors_list:
    #
    #     print("Neighbors: ", n_neighbors)
    #     G = utils.nx_form_unweighted_KNN_graph_indices(indices[:, :n_neighbors + 1])
    #
    #     for i in range(n_repeats):
    #
    #         labels = utils.nx_LPA(G, max_iter = 100)
    #         lpa_ans = getMetric(labels, true_labels)
    #         print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """ Symmetric kNN nx.LPA """
    # n_neighbors = 24
    # n_iter = 100
    # n_threads = 32
    #
    # t1 = timeit.default_timer()
    # G = utils.nx_form_unweighted_sym_KNN_graph(X, k=n_neighbors, n_threads=n_threads)
    # t2 = timeit.default_timer()
    # print('Faiss Time: {}'.format(t2 - t1))
    # labels = utils.nx_LPA(G, max_iter = n_iter)
    # t2 = timeit.default_timer()
    # print('Symmetric kNN LPA Time: {}'.format(t2 - t1))
    #
    # lpa_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """ Symmetric kNN nx.LPA - REPEAT """
    # k_max = 32
    # n_threads = 32
    # indices, distances = utils.faiss_kNN(X, k=k_max + 1, n_threads=n_threads)
    # n_neighbors_list = [12, 16, 20, 24, 28, 32]
    # n_iter = 100
    # for n_neighbors in n_neighbors_list:
    #
    #     print("Neighbors: ", n_neighbors)
    #
    #     G = utils.nx_form_unweighted_sym_KNN_graph_indices(indices[:, :n_neighbors + 1])
    #
    #     for i in range(n_repeats):
    #         labels = utils.nx_LPA(G, max_iter = n_iter)
    #         lpa_ans = getMetric(labels, true_labels)
    #         print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """ Mutual kNN nx.LPA """
    # n_neighbors = 250
    # n_threads = 32
    # n_iter = 100
    # print("Neighbors: ", n_neighbors)
    #
    # t1 = timeit.default_timer()
    # G = utils.nx_form_unweighted_mutual_KNN_graph(X, k=n_neighbors, n_threads=n_threads)
    # t2 = timeit.default_timer()
    # print('Faiss Time: {}'.format(t2 - t1))
    # labels = utils.nx_LPA(G, max_iter=n_iter)
    # t2 = timeit.default_timer()
    # print('Mutual kNN LPA Time: {}'.format(t2 - t1))
    #
    # lpa_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """ Mutual kNN nx.LPA - REPEAT """
    # k_max = 300
    # n_threads = 32
    # indices, distances = utils.faiss_kNN(X, k=k_max + 1, n_threads=n_threads)
    # n_iter = 100
    # n_neighbors_list = [50, 100, 150, 200, 250, 300]
    # for n_neighbors in n_neighbors_list:
    #
    #     print("Neighbors: ", n_neighbors)
    #
    #     G = utils.nx_form_unweighted_mutual_KNN_graph_indices(indices[:, : n_neighbors + 1])
    #
    #     for i in range(n_repeats):
    #         labels = utils.nx_LPA(G, max_iter=n_iter)
    #         lpa_ans = getMetric(labels, true_labels)
    #         print(' '.join(f"{x:.4f}" for x in lpa_ans))

    #============================#

    """====================="""