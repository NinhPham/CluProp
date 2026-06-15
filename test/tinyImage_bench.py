import os

import utils
from utils import getMetric

import sVDC
import clupig
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

    path = "/shared/Dataset/Clustering/"
    # bin_file = path + 'tinyimagenet100_resnet10_embs.bin'
    bin_file = path + 'tinyimagenet100_resnet50_embs.bin'

    n = 98179
    d = 512
    # d = 2048

    true_labels = np.load(path + 'tinyimagenet100_resnet10_labels_98179_512.npy')
    print("Number of true labels: ", len(set(true_labels)))

    n_clusters = 200
    n_iter = 20

    """ Compute Faiss IVFPQ """
    # X = utils.mmap_bin(bin_file, n, d)
    # X = X.astype(np.float32)
    #
    # # Cosine
    # X = normalize(X, norm='l2', axis=1)
    #
    # n_threads = 32
    # nlist = 1024
    # nprobe = 10
    # m = 8
    # dist = "cosine"
    # k_max = 140
    # path = "/shared/Dataset/Clustering/tinyimage100_resnet50_output/"
    #
    # indices, distances = utils.faiss_approx_kNN_IVFPQ(X, k=k_max + 1, n_subquantizer=m, n_list = nlist, n_probe = nprobe, n_threads=n_threads)
    # indices = indices.astype(np.int32)
    # distances = distances.astype(np.float32)
    # del X

    # np.save(path + f"ivfpq_{nlist}_{nprobe}_{m}_{dist}_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int64
    # np.save(path + f"ivfpq_{nlist}_{nprobe}_{m}_{dist}_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32

    """ Compute NNDescent """
    X = utils.mmap_bin(bin_file, n, d)
    X = X.astype(np.float32)

    print("Finish reading data")

    n_threads = 8
    k_max = 10
    savePath = "/shared/Dataset/Clustering/tinyimage100_resnet50_output/"
    seed = 42
    dist="euclidean"

    # NNDescent params
    n_trees = 8
    n_iters = 5
    leafSize = 100
    max_cand = 100

    # Warmup njit
    t1 = timeit.default_timer()



    # It does not count the point itself
    indices, distances = NNDescent(X, n_neighbors=k_max, random_state=seed,
                                   n_trees=n_trees,          # <-- number of RP trees (you choose)
                                   leaf_size=leafSize,        # good rule: ≈ n_neighbors
                                   # max_candidates = max_cand, # "self-join" size of max 50 points
                                   metric=dist, n_iters=n_iters, n_jobs=n_threads).neighbor_graph

    build_time = timeit.default_timer() - t1

    # exact_kNN = np.load(savePath + "exact_Cosine_200_indices.npy").astype(np.int32)
    # r = utils.getAcc_kNNG(indices, exact_kNN[:,:k_max+1])
    print(f"RPT: metric={dist} n_trees={n_trees:2d} n_iters={n_iters:2d} k_max={k_max:2d} max_cand={max_cand:2d} time={build_time:.4f}s")

    indices = indices.astype(np.int32)
    distances = distances.astype(np.float32)
    np.save(savePath + f"nndescent_{n_trees}_{leafSize}_{n_iters}_{dist}_{k_max}_indices.npy", indices)    # shape: (n, k), dtype: int64
    np.save(savePath + f"nndescent_{n_trees}_{leafSize}_{n_iters}_{dist}_{k_max}_distances.npy", distances)  # shape: (n, k), dtype: float32



    """ Simple kNN classification """
    # n_neighbors = 5
    # knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    # knn.fit(X, true_labels)
    #
    # # 5. Predict and evaluate
    # y_pred = knn.predict(X)
    # print("Accuracy:", accuracy_score(true_labels, y_pred))

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

    """ All ig.LPA with precomputed Faiss-IVFPQ/NNDescent/CEOs """

    # indices = np.load(savePath + f"nndescent_{n_trees}_{leafSize}_{n_iters}_{dist}_{k_max}_indices.npy")    # shape: (n, k), dtype: int32
    # distances = np.load(savePath + f"nndescent_{n_trees}_{leafSize}_{n_iters}_{dist}_{k_max}_distances.npy")  # shape: (n, k), dtype: float32

    # Form  graph from indices and distances
    # k_max = 200
    # n_neighbors_list = [12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
    # n_neighbors_list = [10, 15, 20, 25, 30]
    n_neighbors_list = [4, 6, 8, 10, 12, 14, 16]


    c = 1
    print(n_neighbors_list)
    print("c: ", c)

    dbs = clupig.clupig(n, d)
    # dbs.set_min_cluster_size(50)

    for n_neighbors in n_neighbors_list:

        print('n_neighbors: ', n_neighbors)

        # sVDC
        K = min(c * n_neighbors + 1, k_max)

        t1 = timeit.default_timer()
        dbs.dnp_from_knn(indices[:, 1 : K], distances[:, 1 : K], n_neighbors, c=c)
        t2 = timeit.default_timer()
        print('sVDC Time: {}'.format(t2 - t1))

        labels = np.array(dbs.labels_)
        lpa_ans = getMetric(labels, true_labels)
        print(' '.join(f"{x:.4f}" for x in lpa_ans))

        K = min(n_neighbors + 1, k_max)
        # t1 = timeit.default_timer()
        weighted_graph = utils.fast_weighted_sym_knng_igraph(indices[:, 1 : K], distances[:, 1 : K],verbose=True)
        # t2 = timeit.default_timer()
        # print('Graph Construction Time: {}'.format(t2 - t1))

        # Leiden
        t1 = timeit.default_timer()
        labels = utils.run_leiden(weighted_graph)
        t2 = timeit.default_timer()
        print('Leiden Time: {}'.format(t2 - t1))
        lpa_ans = getMetric(labels, true_labels)
        print(' '.join(f"{x:.4f}" for x in lpa_ans))

        # Louvain
        t1 = timeit.default_timer()
        labels = utils.run_louvain(weighted_graph)
        t2 = timeit.default_timer()
        print('Louvain Time: {}'.format(t2 - t1))
        lpa_ans = getMetric(labels, true_labels)
        print(' '.join(f"{x:.4f}" for x in lpa_ans))

    #
    #     del weighted_graph
    #
    #     unweighted_graph = utils.igraph_form_unweighted_sym_KNN_graph(indices[:, :n_neighbors],verbose=True)
    #
    #     # LPA
    #     t1 = timeit.default_timer()
    #     labels = utils.run_LPA(unweighted_graph)
    #     t2 = timeit.default_timer()
    #     print('LPA Time: {}'.format(t2 - t1))
    #     lpa_ans = getMetric(labels, true_labels)
    #     print(' '.join(f"{x:.4f}" for x in lpa_ans))
    #
    #     del unweighted_graph

    """ All ig.LPA with APPROX symmetric kNN FAISS - REPEAT """

    # Form  graph from indices and distances
    # n_neighbors_list = [8, 10, 12, 14, 16, 18, 20]
    # n_neighbors_list = [12, 14, 16, 18, 20, 24, 28, 32, 36, 40]
    # n_neighbors_list = [10, 20]
    # # n_neighbors_list = [65, 70, 75, 80, 85, 90, 95, 100]
    #

    # n_neighbors_list = [10]
    # print(n_neighbors_list)
    #
    # indices = np.load(path + "ivfpq_1024_10_8_Cosine_200_indices.npy")    # shape: (n, k), dtype: int64
    # distances = np.load(path + "ivfpq_1024_10_8_Cosine_200_distances.npy")  # shape: (n, k), dtype: float32
    # indices = indices.astype(np.int32)
    #
    # # print("Shape of the indices:", indices.shape)
    # # print("Shape of the distances:", distances.shape)
    #
    # dbs = sVDC.sVDC(n, d)
    # dbs.set_min_cluster_size(50)
    # c = 5
    # print("c: ", c)
    #
    #
    #
    # for n_neighbors in n_neighbors_list:
    #
    #     print('n_neighbors: ', n_neighbors)
    #
    #
    #     weighted_graph = utils.igraph_form_weighted_sym_KNN_graph_fast(indices[:, :n_neighbors + 1], distances[:, :n_neighbors + 1], use_exp_weight=False, verbose=True)
    #
    #     ## Need more memory for vectorized version
    #     # weighted_graph = utils.fast_igraph_from_faiss(indices[:, :n_neighbors + 1], distances[:, :n_neighbors + 1], use_exp_weight=False, verbose=True)
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

    # del weighted_graph
    #
    #     # unweighted_graph = utils.igraph_form_unweighted_sym_KNN_graph(indices[:, :n_neighbors + 1], verbose=True)
    #     #
    #     # # LPA
    #     # t1 = timeit.default_timer()
    #     # labels = utils.run_LPA(unweighted_graph)
    #     # t2 = timeit.default_timer()
    #     # print('LPA Time: {}'.format(t2 - t1))
    #     # lpa_ans = getMetric(labels, true_labels)
    #     # print(' '.join(f"{x:.4f}" for x in lpa_ans))
    #     #
    #     # del unweighted_graph
    #

    # sVDC
    # t1 = timeit.default_timer()
    # dbs.label_propagation(indices[:, : min(c * n_neighbors, k_max) + 1], distances[:, : min(c * n_neighbors, k_max) + 1], n_neighbors, c=c)
    # t2 = timeit.default_timer()
    # print('sVDC G_k Time: {}'.format(t2 - t1))
    #
    # labels = np.array(dbs.labels_)
    # lpa_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in lpa_ans))
    #
    # t1 = timeit.default_timer()
    # dbs.label_propagation(indices[:, : k_max + 1], distances[:, : k_max + 1], n_neighbors, c=c)
    # t2 = timeit.default_timer()
    # print('sVDC G_k_max Time: {}'.format(t2 - t1))
    #
    # labels = np.array(dbs.labels_)
    # lpa_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in lpa_ans))




    """ All ig.LPA with APPROX symmetric kNN sVDC (L2, Cosine, L1, JS) - REPEAT """
    # X = utils.mmap_bin(bin_file, n, d)
    # X = X.astype(np.float32)
    #
    # # Cosine
    # X = normalize(X, norm='l2', axis=1)
    #
    # n_threads = 32
    #
    # numProj = 512
    # k = 10
    # m = 50
    # topP = 5
    #
    # numEmbed = 1024
    # sigma = 2600 # only used on L1: 16000, L2: 2600
    # dist = "Cosine"
    # clusterNoise = 0 # not used on sOptics
    # output = 'sVDC'
    # numThreads = n_threads
    # verbose = False
    # intervalSampling = 0.4 # only used on Chi2, JS distances
    #
    # seed = -1  # -1 is random
    #
    # dbs = sVDC.sVDC(n, d)
    # dbs.set_params(numProj, k, m, topP, dist, numEmbed, sigma, intervalSampling, verbose, numThreads, seed, output)
    #
    # # n_neighbors_list = [12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
    # # n_neighbors_list = [4, 8, 12, 16, 20, 24]
    #
    # n_neighbors_list = [10, 20, 30, 40, 50]
    # k_max = 500
    # c = 5
    #
    # n_repeats = 1
    # for i in range(n_repeats):
    #
    #     # t1 = timeit.default_timer()
    #     indices, distances = dbs.load_output_akNN(bin_file, k_max)
    #     # t2 = timeit.default_timer()
    #     # print('sVDC form graph time: {}'.format(t2 - t1))
    #
    #     for n_neighbors in n_neighbors_list:
    #
    #         print('n_neighbors: ', n_neighbors)
    #
    #         # sVDC
    #         # t1 = timeit.default_timer()
    #         dbs.label_propagation(indices[:, : c * n_neighbors], distances[:, : c * n_neighbors], n_neighbors, c=c)
    #         # t2 = timeit.default_timer()
    #         # print('sVDC Time: {}'.format(t2 - t1))
    #
    #         labels = np.array(dbs.labels_)
    #         lpa_ans = getMetric(labels, true_labels)
    #         print(' '.join(f"{x:.4f}" for x in lpa_ans))


    # unweighted_graph = utils.igraph_form_unweighted_sym_KNN_graph(indices[:, :n_neighbors])
    #
    # # LPA
    # t1 = timeit.default_timer()
    # labels = utils.run_LPA(unweighted_graph)
    # t2 = timeit.default_timer()
    # # print('LPA Time: {}'.format(t2 - t1))
    # lpa_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in lpa_ans))
    #
    # del unweighted_graph
    #
    #

    # weighted_graph = utils.fast_igraph_from_faiss(indices[:, :n_neighbors], distances[:, :n_neighbors], use_exp_weight=False, verbose=True)
    #
    # # Leiden
    # t1 = timeit.default_timer()
    # labels = utils.run_leiden(weighted_graph)
    # t2 = timeit.default_timer()
    # print('Leiden Time: {}'.format(t2 - t1))
    # lpa_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in lpa_ans))
    #
    # # Louvain
    # t1 = timeit.default_timer()
    # labels = utils.run_louvain(weighted_graph)
    # t2 = timeit.default_timer()
    # print('Louvain Time: {}'.format(t2 - t1))
    # lpa_ans = getMetric(labels, true_labels)
    # print(' '.join(f"{x:.4f}" for x in lpa_ans))

