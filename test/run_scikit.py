import sys
import math
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, OPTICS, KMeans, SpectralClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
import hdbscan

import timeit
#from KernelKMeans import KernelKMeans
# import scipy

def run_scikit_optics():

    # Generate data
    # X, labels_true = make_blobs(
    #     n_samples=10000, n_features=10, centers=10, cluster_std=1, random_state=0
    # )

    X = np.loadtxt('/home/npha145/Dataset/Clustering/mnist_all_X', delimiter=',')
    # X = np.loadtxt('/home/npha145/Dataset/Clustering/mnist_all_X_cosine')
    # labels_true = np.loadtxt('/home/npha145/Dataset/Clustering/mnist_all_y_70K_784')
    print("Finish reading data for optics")



    # Finding top-k distance of sampled points to estimate eps
    # from sklearn.neighbors import NearestNeighbors
    # nbrs = NearestNeighbors(n_neighbors=min_samples, algorithm='ball_tree').fit(X)
    # distances, indexes = nbrs.kneighbors(X)
    #
    # radius = np.mean(distances[:, min_samples - 1])
    # print('Radius: ', radius)

    # OPTICS on Euclidean
    t1 = timeit.default_timer()
    # cosineDist = 0.2
    # eps = math.sqrt(2 * cosineDist)
    eps = 17000
    cluster_D = OPTICS(max_eps=eps, min_samples=50, cluster_method='dbscan', metric='manhattan', algorithm='brute', n_jobs=-1)

    cluster_D.fit(X)
    reachability = cluster_D.reachability_[cluster_D.ordering_]
    t2 = timeit.default_timer()
    print('OPTICS Time: {}'.format(t2 - t1))

    np.savetxt('optics_L1_Eps_17000_MinPts_50_brute_home', np.column_stack((cluster_D.ordering_, reachability)))

    plt.plot(reachability)
    plt.show()

def run_scikit_dbscan():

    # Generate data
    # X, labels_true = make_blobs(
    #     n_samples=10000, n_features=10, centers=10, cluster_std=1, random_state=0
    # )

    X = np.loadtxt('/home/npha145/Dataset/Clustering/covtype_X_cosine')
    y = np.loadtxt('/home/npha145/Dataset/Clustering/covtype_y_581012_54')
    print("Finish reading data for dbscan")

    cosineDist = 0.003
    eps = math.sqrt(2 * cosineDist)
    # eps = 12000
    min_samples = 50
    t1 = timeit.default_timer()
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
    y_pred = db.fit_predict(X)
    t2 = timeit.default_timer()
    print('DBSCAN Time: {}'.format(t2 - t1))

    np.savetxt('dbscan_L0_Eps_0003_MinPts_50', y_pred, delimiter=',')

    print("Number of core points found by Euclidean DBSCAN: {}".format(len(db.core_sample_indices_)))
    print("Acc: Adj. Rand Index Score: %f." % adjusted_rand_score(y_pred, y))
    print("Acc: Adj. Mutual Info Score: %f." % adjusted_mutual_info_score(y_pred, y))
    print("Acc: NMI %f." % normalized_mutual_info_score(y_pred, y))

# hdbscan
def run_hdbscan():
    
    path = "/shared/Dataset/Clustering/"
    dataset = np.loadtxt(path + 'mnist_all_X')

    dataset_t = np.transpose(dataset)
    dataset_t.dtype == np.float32

    y = np.loadtxt(path + 'mnist_all_y_70K_784')

    n, d = np.shape(dataset)
    
    t1 = timeit.default_timer()
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
    y_pred = clusterer.fit_predict(dataset)
    t2 = timeit.default_timer()
    print('HDBSCAN Time: {}'.format(t2 - t1))
    
    print("Acc: Adj. Rand Index Score: %f." % adjusted_rand_score(y_pred, y))
    print("Acc: Adj. Mutual Info Score: %f." % adjusted_mutual_info_score(y_pred, y))
    print("Acc: NMI %f." % normalized_mutual_info_score(y_pred, y))

# Dbscan, k-mean++, kernel k-mean, spectral cluster
def run_scikit_cluster():

    # Generate data
    # X, labels_true = make_blobs(
    #     n_samples=10000, n_features=10, centers=10, cluster_std=1, random_state=0
    # )

    X = np.loadtxt('/home/npha145/Dataset/Clustering/mnist8m_X')
    y = np.loadtxt('/home/npha145/Dataset/Clustering/mnist8m_y_8100000_784')
    print("Finish reading data for clustering")

    """ Dbscan """

    # # cosineDist = 0.003
    # # eps = math.sqrt(2 * cosineDist)
    #
    # eps = 1400
    # min_samples = 50
    # t1 = timeit.default_timer()
    # db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=-1)
    # y_pred = db.fit_predict(X)
    # t2 = timeit.default_timer()
    # print('DBSCAN Time: {}'.format(t2 - t1))
    #
    # np.savetxt('dbscan_L2_Eps_055_MinPts_50', y_pred, delimiter=',')
    #
    # print("Number of core points found by Euclidean DBSCAN: {}".format(len(db.core_sample_indices_)))
    # print("Dbscan: Adj. Rand Index Score: %f." % adjusted_rand_score(y_pred, y))
    # print("Dbscan: Adj. Mutual Info Score: %f." % adjusted_mutual_info_score(y_pred, y))
    # print("Dbscan: NMI %f." % normalized_mutual_info_score(y_pred, y))

    """ kmean++ """
    t1 = timeit.default_timer()
    # kmeans = KMeans(init='k-means++', n_clusters=10, n_init=4, max_iter=10, random_state=0).fit(X)
    kmeans = KMeans(init='random', n_clusters=10, n_init=4, random_state=0).fit(X)
    t2 = timeit.default_timer()
    print('kmean++ Time: {}'.format(t2 - t1))
    y_pred = kmeans.labels_

    np.savetxt('kmeans', y_pred, delimiter=' ')


    print("kmean: Adj. Rand Index Score: %f." % adjusted_rand_score(y_pred, y))
    print("kmean: Adj. Mutual Info Score: %f." % adjusted_mutual_info_score(y_pred, y))
    print("kmean: NMI %f." % normalized_mutual_info_score(y_pred, y))

    """ Kernel kmean"""
    # t1 = timeit.default_timer()
    # km = KernelKMeans(n_clusters=10, max_iter=100, random_state=0, verbose=1).fit(X)
    # t2 = timeit.default_timer()
    # print('kernel mean Time: {}'.format(t2 - t1))
    #
    # y_pred = km.predict(X)
    # np.savetxt('kernel_kmean', y_pred, delimiter=' ')
    #
    # print("Kernel kmean: Adj. Rand Index Score: %f." % adjusted_rand_score(y_pred, y))
    # print("Kernel kmean: Adj. Mutual Info Score: %f." % adjusted_mutual_info_score(y_pred, y))
    # print("Kernel kmean: NMI %f." % normalized_mutual_info_score(y_pred, y))


    """ spectral """
    # t1 = timeit.default_timer()
    # #spectral = SpectralClustering(n_clusters=10, assign_labels='kmeans', random_state=0, gamma=2600, affinity='rbf', n_jobs=-1).fit(X)
    # spectral = SpectralClustering(n_clusters=10, assign_labels='kmeans', random_state=0, affinity='nearest_neighbors', n_jobs=-1).fit(X)
    # t2 = timeit.default_timer()
    # print('Spectral Time: {}'.format(t2 - t1))
    #
    # y_pred = spectral.labels_
    #
    # np.savetxt('spectral', y_pred, delimiter=' ')
    #
    # print("Spectral: Adj. Rand Index Score: %f." % adjusted_rand_score(y_pred, y))
    # print("Spectral: Adj. Mutual Info Score: %f." % adjusted_mutual_info_score(y_pred, y))
    # print("Spectral: NMI %f." % normalized_mutual_info_score(y_pred, y))

def js(p, q):
    p = p + 0.000001
    q = q + 0.000001
    dist = 0
    for i in range(len(p)):
        x = p[i]
        y = q[i]
        dist += (x/2) * math.log2((x + y) / x) + (y/2) * math.log2((x + y) / y)
     
    return 1 - dist

def chi2(p, q):
    p = p + 0.000001
    q = q + 0.000001
    dist = 0
    for i in range(len(p)):
        x = p[i]
        y = q[i]
        dist += 2 * (x * y) / (x + y)

    return 1 - dist

def run_scikit_dbscan_JS():

    X = np.loadtxt('/home/npha145/Dataset/Clustering/mnist_all_X_prob')
    y = np.loadtxt('/home/npha145/Dataset/Clustering/mnist_all_y_70K_784')
    print("Finish reading data for clustering")

    eps_gap = 0.01
    eps_start = 0.12
    min_samples = 50
    s = 5

    for j in range(s):    

        eps = eps_start + eps_gap * j

        t1 = timeit.default_timer()
        db = DBSCAN(eps=eps, min_samples=min_samples, metric=js, n_jobs=-1, algorithm='brute')
        y_pred = db.fit_predict(X)
        t2 = timeit.default_timer()
        print('Scikit Dbscan Time: {}'.format(t2 - t1))

        filename = 'mnistAll_Prob/scikitDbscan_L4_' + str(round(eps * 1000))
        np.savetxt(filename, y_pred, delimiter=' ')

        # print("Number of core points found by Euclidean DBSCAN: {}".format(len(db.core_sample_indices_)))
        # print("Dbscan: Adj. Rand Index Score: %f." % adjusted_rand_score(y_pred, y))
        # print("Dbscan: Adj. Mutual Info Score: %f." % adjusted_mutual_info_score(y_pred, y))
        print("Dbscan: NMI %f." % normalized_mutual_info_score(y_pred, y))


# main
run_hdbscan()