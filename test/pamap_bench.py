import os

import utils
from utils import getMetric

import clupig
import faiss

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

    path = "/shared/Dataset/Clustering/"
    savePath = "/shared/Dataset/Clustering/pamap2_output/"

    n = 1770131
    d = 51
    bin_file = path + 'pamap2_X_no_0.bin'

    X = utils.mmap_bin(path + 'pamap2_X_no_0.bin', n, d)
    X.dtype == np.float32

    # Normalize in case need it
    # X = normalize(X, norm='l2', axis=1)
    # X = normalize(X, norm='l1', axis=1)
    # X /= X.sum(axis=1, keepdims=True) # Normalize each row to sum to 1 (L1 normalization)

    # nan_mask = np.isnan(X)
    # print(f"NaN mask: {nan_mask}")
    #
    # nan_indices = np.where(nan_mask)
    # print(f"Indices of NaN values: {nan_indices}")

    true_labels = np.loadtxt(path + 'pamap2_y_no_0_1770131_51', dtype=np.int32)
    n_clusters = 18
    n_iter = 100
    n_repeats = 5
    n_threads = 32

    """====================="""

    """ Compute exact kNN """
    # n_threads = 32
    # k_max = 200
    #
    # # Exact L2
    # indices, distances = utils.faiss_kNN(X, k=k_max + 1, n_threads=n_threads)
    # np.save(savePath + "exact_L2_200_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath + "exact_L2_200_distances.npy", distances)  # shape: (n, k), dtype: float32
    #
    # # Exact L1
    # nbrs = NearestNeighbors(n_neighbors=k_max + 1, metric='manhattan',n_jobs=n_threads).fit(X)
    # distances, indices = nbrs.kneighbors(X)
    # np.save(savePath + "exact_L1_200_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath + "exact_L1_200_distances.npy", distances)  # shape: (n, k), dtype: float32
    #
    # # Exact Cosine
    # X = normalize(X, norm='l2', axis=1)
    # indices, distances = utils.faiss_kNN(X, k=k_max + 1, n_threads=n_threads)
    # np.save(savePath + "exact_Cosine_200_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath + "exact_Cosine_200_distances.npy", distances)  # shape: (n, k), dtype: float32

    """ Compute Faiss approx kNN (IVF and IVFPQ) """
    # n_threads = 32
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
    # np.save(savePath + "ivf_512_10_L2_500_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath + "ivf_512_10_L2_500_distances.npy", distances)  # shape: (n, k), dtype: float32
    #
    # indices, distances = utils.faiss_approx_kNN_IVFPQ(X, k=k_max + 1, n_subquantizer=m, n_list = nlist, n_probe = nprobe, n_threads=n_threads)
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    # np.save(savePath + "ivfpq_512_10_3_L2_500_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath + "ivfpq_512_10_3_L2_500_distances.npy", distances)  # shape: (n, k), dtype: float32
    #
    # # Cosine
    # X = normalize(X, norm='l2', axis=1)
    #
    # indices, distances = utils.faiss_approx_kNN_IVF(X, k=k_max + 1, n_list = nlist, n_probe = nprobe, n_threads=n_threads)
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    # np.save(savePath + "ivf_512_10_Cosine_500_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath + "ivf_512_10_Cosine_500_distances.npy", distances)  # shape: (n, k), dtype: float32
    #
    # indices, distances = utils.faiss_approx_kNN_IVFPQ(X, k=k_max + 1, n_subquantizer=m, n_list = nlist, n_probe = nprobe, n_threads=n_threads)
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    # np.save(savePath + "ivfpq_512_10_3_Cosine_500_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath + "ivfpq_512_10_3_Cosine_500_distances.npy", distances)  # shape: (n, k), dtype: float32

    """ Compute NNDescent """
    n_threads = 8
    k_max = 20
    seed = 42
    dist="euclidean"
    #
    # # NNDescent params
    n_trees = 8
    n_iters = 1
    # leafSize = int(k / n_trees)
    leafSize = k_max

    t1 = timeit.default_timer()

    # Warmup njit

    # NNDescent(X, n_neighbors=k_max, random_state=seed, tree_init=True,
    #           # init_graph=indices,
    #           metric=dist, n_iters=1, n_jobs=n_threads)

    # It does not count the point itself
    indices, distances = NNDescent(X, n_neighbors=k_max, random_state=None,
                                   n_trees=n_trees,          # <-- number of RP trees (you choose)
                                   leaf_size=leafSize,        # good rule: ≈ n_neighbors
                                   metric=dist, n_iters=n_iters, n_jobs=n_threads).neighbor_graph

    build_time = timeit.default_timer() - t1

    print(f"Beelink-8: RPT: metric={dist} n_trees={n_trees:2d} n_iters={n_iters:2d} leafSize={leafSize:2d} k_max={k_max:2d} time={build_time:.4f}s")

    indices = indices.astype(np.int32)
    distances = distances.astype(np.float32)

    # np.save(savePath + f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(savePath + f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32

    """====================="""

    """ faiss k-mean """
    # n_threads = 8
    # n_iter = 40
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
    # t1 = timeit.default_timer()
    # metric = "euclidean"
    # labels = utils.spectral_clustering(
    #     X,
    #     n_clusters=n_clusters,
    #     k=50,
    #     metric=metric,
    #     mutual=True,          # try False for symmetric-kNN
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
    # # There is -1 as it is not not connected to the largest component, need to increase k
    # vals, counts = np.unique(labels, return_counts=True)
    # for v, c in zip(vals, counts):
    #     print(f"{v}: {c}")

    """ Nystrom kernel kmean++ """
    # Compute pairwise Euclidean distances over Subsample to avoid O(n^2) for large MNIST
    # n_samples = 1000
    # X_sample = X[np.random.choice(len(X), n_samples, replace=False)]
    # dists = pairwise_distances(X_sample, metric="euclidean")
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
    # dists = pairwise_distances(X_sample, metric="euclidean")
    # median_dist = np.median(dists)
    # #
    # # Recommended gamma:
    # gamma = 1 / (2 * median_dist ** 2)
    #
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

    """====================="""

    """ iGraph propagation with precomputed Faiss/NNDescent symmetric kNN (need +1 as Faiss consider the point itself as part of kNN) """
    n_threads = 8


    # # indices = np.load(path + "ivfpq_512_10_3_L2_500_indices.npy")    # shape: (n, k), dtype: int64
    # # distances = np.load(path + "ivfpq_512_10_3_L2_500_distances.npy")  # shape: (n, k), dtype: float32
    # # indices = np.load(path + "ivf_512_10_L2_500_indices.npy")    # shape: (n, k), dtype: int64
    # # distances = np.load(path + "ivf_512_10_L2_500_distances.npy")  # shape: (n, k), dtype: float32
    #
    # # # NNDescent params
    # n_iters = 1
    # n_trees = 16
    # k_max = 20
    # leafSize = k_max
    # dist="euclidean"
    #
    # indices = np.load(path + f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_indices.npy")    # shape: (n, k), dtype: int64
    # distances = np.load(path + f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_distances.npy")  # shape: (n, k), dtype: float32
    #
    # # n_neighbors_list = [4, 8, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48]
    # # n_neighbors_list = [10, 12, 14, 16, 18, 20]
    # # n_neighbors_list = [2, 3, 4, 5, 6]

    # n_neighbors_list = [10]
    #
    # print(n_neighbors_list)
    #
    # for n_neighbors in n_neighbors_list:
    #
    #     print('n_neighbors: ', n_neighbors)
    #
    #     #
    #     # # LPA
    #     # unweighted_graph = utils.fast_unweighted_sym_knng_igraph(indices[:, :n_neighbors], verbose = True)
    #     # t1 = timeit.default_timer()
    #     # labels = utils.run_LPA(unweighted_graph)
    #     # t2 = timeit.default_timer()
    #     # print('LPA Time: {}'.format(t2 - t1))
    #     # lpa_ans = getMetric(labels, true_labels)
    #     # print(' '.join(f"{x:.4f}" for x in lpa_ans))
    #     # del unweighted_graph
    #
    #     # NNDescent: need  1: as it count the point itself
    #     # Faiss: need  1: n_neighbor+1 as it count the point itself
    #     K = min(n_neighbors+1, k_max)
    #
    #     # symmetric kNN
    #     weighted_graph = utils.fast_weighted_sym_knng_igraph(indices[:, 1:K], distances[:, 1:K], use_exp_weight=False, verbose = True)
    #     # weighted_graph = utils.fast_weighted_sym_knng_igraph_large_mem(indices[:, 1:K], distances[:, 1:K], use_exp_weight=False, verbose = True)
    #
    #
    #     # Leiden
    #     t1 = timeit.default_timer()
    #     labels = utils.run_leiden(weighted_graph)
    #     t2 = timeit.default_timer()
    #     print('Leiden w = 1/d Time: {}'.format(t2 - t1))
    #     lpa_ans = getMetric(labels, true_labels)
    #     print(' '.join(f"{x:.4f}" for x in lpa_ans))

        # Louvain
        # t1 = timeit.default_timer()
        # labels = utils.run_louvain(weighted_graph)
        # t2 = timeit.default_timer()
        # print('Louvain w = 1/d Time: {}'.format(t2 - t1))
        # lpa_ans = getMetric(labels, true_labels)
        # print(' '.join(f"{x:.4f}" for x in lpa_ans))


        # weighted_graph = utils.fast_weighted_sym_knng_igraph(indices[:, :n_neighbors + 1], distances[:, :n_neighbors + 1], use_exp_weight=True, verbose = True)
        #
        # # Leiden
        # t1 = timeit.default_timer()
        # labels = utils.run_leiden(weighted_graph)
        # t2 = timeit.default_timer()
        # print('Leiden w = exp(-d) Time: {}'.format(t2 - t1))
        # lpa_ans = getMetric(labels, true_labels)
        # print(' '.join(f"{x:.4f}" for x in lpa_ans))
        #
        # # Louvain
        # t1 = timeit.default_timer()
        # labels = utils.run_louvain(weighted_graph)
        # t2 = timeit.default_timer()
        # print('Louvain w = exp(-d) Time: {}'.format(t2 - t1))
        # lpa_ans = getMetric(labels, true_labels)
        # print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """ iGraph propagation with precomputed CEOs (Cosine, L2, L1) - REPEAT """
    """ CEOs does not contain the point itself as part of kNN, so no need +1 """
    # n_threads = 32
    # k_max = 500
    #
    # # n_neighbors_list = [12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
    # # n_neighbors_list = [16, 20, 24, 28, 32]
    # n_neighbors_list = [4, 8, 12, 14, 16, 20]
    # # n_neighbors_list = [28, 32, 36, 40, 44, 48, 52, 56, 60]
    # n_neighbors_list = [12]
    # n_repeats = 1
    #
    # for i in range(n_repeats):
    #
    #     indices = np.load(savePath + f"ceos2_512_20_50_5_L2_500_indices_{i+1}.npy")    # shape: (n, k), dtype: int64
    #     distances = np.load(savePath + f"ceos2_512_20_50_5_L2_500_distances_{i+1}.npy")  # shape: (n, k), dtype: float32
    #
    #     # print(indices[0:1, :20])
    #     # print(distances[0:1, :20])
    #     for n_neighbors in n_neighbors_list:
    #
    #         print('n_neighbors: ', n_neighbors)
    #
    #         # # Note: LPA runs very slow on unweighted graph even for n_neighbors = 4, so skip it here
    #         unweighted_graph = utils.fast_unweighted_sym_knng_igraph(indices[:, :n_neighbors], verbose = True)
    #
    #         # LPA
    #         t1 = timeit.default_timer()
    #         labels = utils.run_LPA(unweighted_graph)
    #         t2 = timeit.default_timer()
    #         print('LPA Time: {}'.format(t2 - t1))
    #         lpa_ans = getMetric(labels, true_labels)
    #         print(' '.join(f"{x:.4f}" for x in lpa_ans))
    #
    #         del unweighted_graph
    #
    #         weighted_graph = utils.fast_weighted_sym_knng_igraph(indices[:, :n_neighbors], distances[:, :n_neighbors], use_exp_weight=False, verbose = True)
    #
    #         # Leiden
    #         t1 = timeit.default_timer()
    #         labels = utils.run_leiden(weighted_graph)
    #         t2 = timeit.default_timer()
    #         print('Leiden Time: {}'.format(t2 - t1))
    #         lpa_ans = getMetric(labels, true_labels)
    #         print(' '.join(f"{x:.4f}" for x in lpa_ans))
    #
    #         # Louvain
    #         t1 = timeit.default_timer()
    #         labels = utils.run_louvain(weighted_graph)
    #         t2 = timeit.default_timer()
    #         print('Louvain Time: {}'.format(t2 - t1))
    #         lpa_ans = getMetric(labels, true_labels)
    #         print(' '.join(f"{x:.4f}" for x in lpa_ans))
    #
    #         del weighted_graph

    """====================="""
    """ (c,k)-DNP with precomputed EXACT/Faiss/NNDescent symmetric kNN """
    """ c > 1 gives higher accuracy, and G_kmax where kmax > c*k gives more stable accuracy than G_k """
    n_threads = 8
    # n_trees = 8
    # n_iters = 1
    # k_max = 10
    # leafSize = 100
    # dist="euclidean"

    # indices = np.load(path + "ivf_512_10_L2_500_indices.npy")    # shape: (n, k), dtype: int64
    # distances = np.load(path + "ivf_512_10_L2_500_distances.npy")  # shape: (n, k), dtype: float32
    # indices = np.load(path + "ivfpq_512_10_3_L2_500_indices.npy")    # shape: (n, k), dtype: int64
    # distances = np.load(path + "ivfpq_512_10_3_L2_500_distances.npy")  # shape: (n, k), dtype: float32
    # indices = np.load(path + f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_indices.npy")    # shape: (n, k), dtype: int64
    # distances = np.load(path + f"nndescent_{n_iters}_{n_trees}_{leafSize}_{dist}_{k_max}_distances.npy")  # shape: (n, k), dtype: float32

    # print('shape of array :', indices.shape)
    # print(indices[0, 0 : 10])
    # print(distances[0, 0 : 10])

    dbs = clupig.clupig(n, d)
    dbs.set_threads(n_threads)


    # n_neighbors_list = [4, 8, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48]
    # n_neighbors_list = [4, 8, 12, 14, 16, 20, 24, 28, 32]
    # n_neighbors_list = [5, 10, 15, 20, 25]
    # n_neighbors_list = [ 25, 30, 35, 40, 45, 50]
    # n_neighbors_list = [int(x * 2) for x in n_neighbors_list]
    # n_neighbors_list = [30, 40, 50, 60, 70, 80]
    # n_neighbors_list = [90, 100, 120, 130, 140, 150]
    # n_neighbors_list = [10, 12, 14, 16, 18, 20]
    n_neighbors_list = [20]

    # Larger K is not useful since we have minConnectedDist to limit the propagation.
    # This is because points will be added in PQ if having smaller minConnectionDist
    # K = 12

    # this param will significantly increase number of clusters (i.e. identifying more noise clusters)
    # dbs.set_propagation_cutoff()

    c = 1
    print(n_neighbors_list)
    print("c: ", c)

    for n_neighbors in n_neighbors_list:

        print('n_neighbors: ', n_neighbors) # k' in the paper

        # clupig
        t1 = timeit.default_timer()

        # G_K where K = ck
        # NNDescent and Faiss consider the point itself, so need to start from 1
        K = min(c * n_neighbors + 1, k_max)
        dbs.dnp_from_knn(indices[:, 1 : K], distances[:, 1 : K], n_neighbors, c=c)
        lpa_ans = getMetric(np.array(dbs.labels_), true_labels)
        print(' '.join(f"{x:.4f}" for x in lpa_ans))

        t2 = timeit.default_timer()
        print('sVDC Time: {}'.format(t2 - t1))

        # G_kmax where ck <= K_max,
        # K = k_max
        #
        # t1 = timeit.default_timer()
        # dbs.dnp_from_knn(indices[:, 1 : K], distances[:, 1 : K], n_neighbors, c=c)
        # lpa_ans = getMetric(np.array(dbs.labels_), true_labels)
        # print(' '.join(f"{x:.4f}" for x in lpa_ans))
        #
        # t2 = timeit.default_timer()
        # print('sVDC Time: {}'.format(t2 - t1))

    """ (c,k)-DNP with precomputed EXACT/Faiss symmetric kNN """
    """ c > 1 gives higher accuracy, and G_kmax where kmax > c*k gives more stable accuracy than G_k """
    # n_threads = 32
    #
    # # Load precompute kNNG
    # savePath = "/shared/Dataset/Clustering/mnist70K_output/"
    # indices = np.load(savePath + "ivf_100_10_Cosine_200_indices.npy")    # shape: (n, k), dtype: int64
    # distances = np.load(savePath + "ivf_100_10_Cosine_200_distances.npy")  # shape: (n, k), dtype: float32
    #
    # # n_neighbors_list = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    # # n_neighbors_list = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    # n_neighbors_list = [4, 5, 6, 7, 8, 9, 10]
    # # n_neighbors_list = [8]
    # # n_neighbors_list = [8, 10, 12, 14, 16, 18, 20]
    #
    # k_max = 40
    # c = 2
    # dbs = clupig.clupig(n, d)
    # # dbs.set_min_cluster_size(50)
    #
    # for n_neighbors in n_neighbors_list:
    #
    #     print('n_neighbors: ', n_neighbors) # k' in the paper
    #
    #     # clupig
    #     t1 = timeit.default_timer()
    #     # G_K where K = ck
    #     dbs.DNP_from_kNN(indices[:, : min(c * n_neighbors, k_max) + 1], distances[:, : min(c * n_neighbors, k_max) + 1], n_neighbors, c=c)
    #     # G_kmax where ck <= K_max,
    #     # dbs.DNP_from_kNN(indices[:, : k_max + 1], distances[:, : k_max + 1], n_neighbors, c=c)
    #
    #     t2 = timeit.default_timer()
    #     # print('clupig Time: {}'.format(t2 - t1))
    #
    #     labels = np.array(dbs.labels_)
    #     lpa_ans = getMetric(labels, true_labels)
    #     print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """====================="""

    """ kNN LPA """
    # n_neighbors = 24
    # n_threads = 32
    # print("Neighbors: ", n_neighbors)
    # t1 = timeit.default_timer()
    # G = utils.build_knn_graph_faiss(X, k=n_neighbors, n_threads=n_threads)
    # t2 = timeit.default_timer()
    # print('Faiss Time: {}'.format(t2 - t1))
    # labels = label_propagation(G) # return [ [1, 3, 5], [2, 4, 6], [10, 11, 7, 8, 9] ], each list is a cluster
    # t2 = timeit.default_timer()
    # print('kNN LPA Time: {}'.format(t2 - t1))
    #
    # lpa_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """ kNN LPA - REPEAT """
    # n_neighbors_list = [12, 16, 20, 24, 28, 32]
    # n_threads = 8
    #
    # for n_neighbors in n_neighbors_list:
    #
    #     print("Neighbors: ", n_neighbors)
    #     G = utils.build_knn_graph_faiss(X, k=n_neighbors, n_threads=n_threads)
    #
    #     for i in range(n_repeats):
    #
    #         labels = utils.label_propagation(G) # return [ [1, 3, 5], [2, 4, 6], [10, 11, 7, 8, 9] ], each list is a cluster
    #         lpa_ans = getMetric(labels, true_labels)
    #         print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """ Symmetric kNN LPA """
    # n_neighbors = 24
    # n_iter = 100
    # n_threads = 32
    #
    # t1 = timeit.default_timer()
    # G = utils.build_symmetric_knn_graph_faiss(X, k=n_neighbors, n_threads=n_threads)
    # t2 = timeit.default_timer()
    # print('Faiss Time: {}'.format(t2 - t1))
    # labels = utils.label_propagation(G, max_iter=n_iter) # return [ [1, 3, 5], [2, 4, 6], [10, 11, 7, 8, 9] ], each list is a cluster
    # t2 = timeit.default_timer()
    # print('Symmetric kNN LPA Time: {}'.format(t2 - t1))
    #
    # lpa_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """ Symmetric LPA - REPEAT """
    # n_neighbors_list = [12, 16, 20, 24, 28, 32]
    # n_iter = 100
    # n_threads = 8
    # for n_neighbors in n_neighbors_list:
    #
    #     print("Neighbors: ", n_neighbors)
    #
    #     G = utils.build_symmetric_knn_graph_faiss(X, k=n_neighbors, n_threads=n_threads)
    #
    #     for i in range(n_repeats):
    #         labels = utils.label_propagation(G, max_iter=n_iter) # return [ [1, 3, 5], [2, 4, 6], [10, 11, 7, 8, 9] ], each list is a cluster
    #         lpa_ans = getMetric(labels, true_labels)
    #         print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """ Mutual kNN LPA """
    # n_neighbors = 24
    # n_threads = 32
    # print("Neighbors: ", n_neighbors)
    #
    # t1 = timeit.default_timer()
    # G = utils.build_mutual_knn_graph_faiss(X, k=n_neighbors, n_threads=n_threads)
    # t2 = timeit.default_timer()
    # print('Faiss Time: {}'.format(t2 - t1))
    # labels = utils.label_propagation(G)
    # t2 = timeit.default_timer()
    # print('Mutual kNN LPA Time: {}'.format(t2 - t1))
    #
    # lpa_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """ Mutual kNN LPA - REPEAT """
    # n_threads = 8
    # n_neighbors_list = [50, 100, 150, 200, 250, 300]
    # for n_neighbors in n_neighbors_list:
    #
    #     print("Neighbors: ", n_neighbors)
    #
    #     G = utils.build_mutual_knn_graph_faiss(X, k=n_neighbors, n_threads=n_threads)
    #
    #     for i in range(n_repeats):
    #         labels = utils.label_propagation(G)
    #         lpa_ans = getMetric(labels, true_labels)
    #         print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """ Faiss symmetric LPA """
    # n_neighbors = 50
    # n_iter = 20
    # n_threads = 32
    # nlist = 4096      # partition into 4096 clusters
    # nprobe = 64       # probe 64 clusters at query time
    #
    # t1 = timeit.default_timer()
    # G = utils.build_approx_symmetric_knn_graph_faiss(X, k=n_neighbors, n_list = nlist, n_probe = nprobe, n_threads=n_threads)
    # t2 = timeit.default_timer()
    # print('Faiss Time: {}'.format(t2 - t1))
    #
    # labels = utils.label_propagation(G, max_iter=n_iter)
    # t2 = timeit.default_timer()
    # print('Faiss LPA Time: {}'.format(t2 - t1))
    #
    # lpa_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """ Faiss symmetric LPA - REPEAT """

    # n_neighbors_list = [50]
    # n_iter = 20
    # n_threads = 8
    # nlist = 4096      # partition into 4096 clusters
    # nprobe = 64       # probe 64 clusters at query time
    #
    # for n_neighbors in n_neighbors_list:
    #
    #     print("Neighbors: ", n_neighbors)
    #     G = utils.build_approx_symmetric_knn_graph_faiss(X, k=n_neighbors, n_list = nlist, n_probe = nprobe, n_threads=n_threads)
    #
    #     for i in range(n_repeats):
    #
    #         labels = utils.label_propagation(G, max_iter=n_iter)
    #         lpa_ans = getMetric(labels, true_labels)
    #         print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """ Scann symmetric kNN LPA """
    # n_neighbors = 50
    # n_iter = 20
    # n_threads = 32
    # nlist = 4096      # partition into 4096 clusters
    # nprobe = 64       # probe 64 clusters at query time
    #
    # t1 = timeit.default_timer()
    # G = utils.build_approx_symmetric_knn_graph_scann(X, k=n_neighbors, n_list = nlist, n_probe = nprobe, n_threads=n_threads)
    # t2 = timeit.default_timer()
    # print('SCANN Time: {}'.format(t2 - t1))
    #
    # labels = utils.label_propagation(G, max_iter=n_iter)
    #
    # t2 = timeit.default_timer()
    # print('SCANN LPA Time: {}'.format(t2 - t1))
    #
    # lpa_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """ Scann symmetric kNN LPA - REPEAT """
    # n_neighbors_list = [50]
    # n_iter = 20
    # n_threads = 8
    # nlist = 4096      # partition into 4096 clusters
    # nprobe = 64       # probe 64 clusters at query time
    #
    # for n_neighbors in n_neighbors_list:
    #
    #     print("Neighbors: ", n_neighbors)
    #
    #     G = utils.build_approx_symmetric_knn_graph_scann(X, k=n_neighbors, n_list = nlist, n_probe = nprobe, n_threads=n_threads)
    #
    #     for i in range(n_repeats):
    #
    #         labels = utils.label_propagation(G, max_iter=n_iter)
    #         lpa_ans = getMetric(labels, true_labels)
    #         print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """ igraph with scann"""
    # n_neighbors = 100
    # n_threads = 32
    # nlist = 4096      # partition into 4096 clusters
    # nprobe = 64       # probe 64 clusters at query time
    #
    # t1 = timeit.default_timer()
    # labels = utils.label_propagation_from_scann(X, k=n_neighbors, n_list = nlist, n_probe = nprobe, n_threads=32)
    # t2 = timeit.default_timer()
    # print('Scann iGraph Time: {}'.format(t2 - t1))
    #
    # lpa_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """ All LPA with symmetric kNN - REPEAT """

    # X = X.astype(np.float32)
    # n, d = X.shape
    # n_threads = 32
    #
    # # X = normalize(X, norm='l2', axis=1)
    #
    # # Cosine or L2
    # # 1. Create FAISS index
    # faiss.omp_set_num_threads(n_threads) # This is also default
    # nlist = 512 #4096  # the number of clusters
    # nprobe = 10 # 64
    #
    # k_max = 100
    # indices, distances = utils.faiss_approx_kNN(X, k=k_max + 1, n_list = nlist, n_probe = nprobe, n_threads=n_threads)
    #
    # dbs = clupig.clupig(n, d)
    # c = 1
    #
    # # Form  graph from indices and distances
    # # unweighted_graph = utils.igraph_form_unweighted_sym_KNN_graph(indices)
    # weighted_graph = utils.igraph_form_weighted_sym_KNN_graph(indices, distances)
    #
    # n_neighbors_list = [50, 100]
    #
    # for n_neighbors in n_neighbors_list:
    #
    #     print('n_neighbors: ', n_neighbors)
    #
    #     # clupig
    #     t1 = timeit.default_timer()
    #     dbs.label_propagation(indices[:, :n_neighbors + 1].tolist(), distances[:, :n_neighbors + 1].tolist(), n_neighbors, c=c)
    #     t2 = timeit.default_timer()
    #     print('clupig Time: {}'.format(t2 - t1))
    #
    #     labels = np.array(dbs.labels_)
    #     lpa_ans = getMetric(labels, true_labels)
    #     print(' '.join(f"{x:.4f}" for x in lpa_ans))
    #
    #     # LPA
    #     # t1 = timeit.default_timer()
    #     # labels = utils.run_LPA(unweighted_graph)
    #     # t2 = timeit.default_timer()
    #     # print('LPA Time: {}'.format(t2 - t1))
    #     # lpa_ans = getMetric(labels, true_labels)
    #     # print(' '.join(f"{x:.4f}" for x in lpa_ans))
    #
    #     # Leiden
    #     t1 = timeit.default_timer()
    #     labels = utils.run_leiden(weighted_graph)
    #     t2 = timeit.default_timer()
    #     print('Leiden Time: {}'.format(t2 - t1))
    #     lpa_ans = getMetric(labels, true_labels)
    #     print(' '.join(f"{x:.4f}" for x in lpa_ans))
    #
    #     # Louvain
    #     t1 = timeit.default_timer()
    #     labels = utils.run_louvain(weighted_graph)
    #     t2 = timeit.default_timer()
    #     print('Louvain Time: {}'.format(t2 - t1))
    #     lpa_ans = getMetric(labels, true_labels)
    #     print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """ clupig LPA with approx symmetric kNN Faiss - REPEAT """

    # X = X.astype(np.float32)
    # n, d = X.shape
    #
    # n_threads = 32
    # nlist = 4096
    # nprobe = 64
    # k_max = 200
    # n_neighbors_list = [110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    #
    # # L2
    # faiss.omp_set_num_threads(n_threads) # This is also default
    # t1 = timeit.default_timer()
    # quantizer = faiss.IndexFlatL2(d)  # the other index
    # index = faiss.IndexIVFFlat(quantizer, d, nlist)
    # index.nprobe = nprobe
    # # 8 specifies that each sub-vector is encoded as 8 bits
    # index.train(X)
    # index.add(X)
    #
    # t2 = timeit.default_timer()
    # print('Construction time of Faiss IVF: {}'.format(t2 - t1))
    #
    # # 2. Perform search (note: k+1 because the closest is the point itself)
    # distances, indices = index.search(X, k_max + 1)
    # t2 = timeit.default_timer()
    # print('Build and query time of Faiss IVF: {}'.format(t2 - t1))
    #
    #
    # dbs = clupig.clupig(n, d)
    # c = 1
    #
    # for n_neighbors in n_neighbors_list:
    #
    #     # 2. Perform search (note: k+1 because the closest is the point itself)
    #     dbs.label_propagation(indices[:, :n_neighbors + 1].tolist(), distances[:, :n_neighbors + 1].tolist(), n_neighbors, c=c)
    #     labels = np.array(dbs.labels_)
    #     lpa_ans = getMetric(labels, true_labels)
    #     print(' '.join(f"{x:.4f}" for x in lpa_ans))

    #
    # # Cosine
    # X = normalize(X, norm='l2', axis=1)
    # n_threads = 32
    # nlist = 4096
    # nprobe = 64
    # k_max = 200
    # n_neighbors_list = [110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    #
    # faiss.omp_set_num_threads(n_threads) # This is also default
    # t1 = timeit.default_timer()
    # quantizer = faiss.IndexFlatL2(d)  # the other index
    # index = faiss.IndexIVFFlat(quantizer, d, nlist)
    # index.nprobe = nprobe
    # # 8 specifies that each sub-vector is encoded as 8 bits
    # index.train(X)
    # index.add(X)
    #
    # t2 = timeit.default_timer()
    # print('Construction time of Faiss IVF: {}'.format(t2 - t1))
    #
    # # 2. Perform search (note: k+1 because the closest is the point itself)
    # distances, indices = index.search(X, k_max + 1)
    # t2 = timeit.default_timer()
    # print('Build and query time of Faiss IVF: {}'.format(t2 - t1))
    #
    #
    # dbs = clupig.clupig(n, d)
    # c = 1
    #
    # for n_neighbors in n_neighbors_list:
    #
    #     # 2. Perform search (note: k+1 because the closest is the point itself)
    #     dbs.label_propagation(indices[:, :n_neighbors + 1].tolist(), distances[:, :n_neighbors + 1].tolist(), n_neighbors, c=c)
    #     labels = np.array(dbs.labels_)
    #     lpa_ans = getMetric(labels, true_labels)
    #     print(' '.join(f"{x:.4f}" for x in lpa_ans))


    """ Manual distance computation (L1 and JS) LPA - REPEAT """
    # k_max = 14
    # knn_indices = mmap_bin(path + 'mnist_all_X_kNN_L1_14.bin', n, k_max)
    # n_neighbors_list = [2, 4, 6, 8, 10, 12, 14]

    # k_max = 32
    # knn_indices = mmap_bin(path + 'mnist_all_X_kNN_JS_32.bin', n, k_max)
    # n_neighbors_list = [12, 16, 20, 24, 28, 32]

    # for n_neighbors in n_neighbors_list:
    #
    #     print("Neighbors: ", n_neighbors)
    #
    #     # 3. Build undirected graph
    #     edges = set()
    #
    #     for i in range(n):
    #         for j in indices[i]:
    #             if i != j:
    #                 edge = tuple(sorted((i, j)))  # ensures (i,j) and (j,i) treated as same
    #                 edges.add(edge)
    #
    #     G = nx.Graph()
    #     G.add_edges_from(edges)
    #     print("Number of nodes: ", G.number_of_nodes())
    #
    #     for i in range(n_repeats):
    #         labels = utils.label_propagation(G)
    #         lpa_ans = getMetric(labels, true_labels)
    #         print(' '.join(f"{x:.4f}" for x in lpa_ans))

    """ Faiss LPA with n_clusters """
    # n_neighbors = 12
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

    """====================="""

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

    """ Hdbscan """
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
    # eps = 20000
    # minPts = 12
    # run_sOptics(X, minPts, eps)

    """ sDbscan"""
    # dist = "L1"
    # for i in range(5):
    #     minPts_list = [2, 3, 4, 5, 6, 7]
    #     eps_list = [5000, 6000, 7000, 8000, 9000, 10000]
    #     for minPts in minPts_list:
    #         print("minPts: ", minPts)
    #         for eps in eps_list:
    #             ans = run_sDbscan(X, minPts, eps, dist)
    #             print(' '.join(f"{val:.3f}" for val in ans))

    # run_sDbscan(X, minPts=24, eps=0.13, dist="Cosine", n_threads = 32)

    """ sngDbscan"""
    # dist = "Cosine"
    # for i in range(5):
    #     minPts_list = [12, 16, 20, 24, 28, 32]
    #     eps_list = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
    #     for minPts in minPts_list:
    #         print("minPts: ", minPts)
    #         for eps in eps_list:
    #             ans = run_sngDbscan(X, minPts, eps, dist)
    #             print(' '.join(f"{val:.3f}" for val in ans))

    # run_sngDbscan(X, minPts=24, eps=0.13, dist="Cosine", n_threads = 32)

    """====================="""


