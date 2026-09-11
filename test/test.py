import numpy as np
import cluprop
from pynndescent import NNDescent
from sklearn.datasets import fetch_openml
import timeit
import utils

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False
)

X = mnist.data
y = mnist.target.astype(int)

print(X.shape)  # (70000, 784)
print(y.shape)  # (70000,)

n_threads = 8
k_max = 20
dist = "cosine"

# NNDescent params
n_trees = 8
n_iters = 5
leafSize = 50

t1 = timeit.default_timer()
indices, distances = NNDescent(X, n_neighbors=k_max, random_state=None,
                               n_trees=n_trees,          # <-- number of RP trees (you choose)
                               leaf_size=leafSize,        # good rule: ≈ n_neighbors
                               metric=dist, n_iters=n_iters, n_jobs=n_threads).neighbor_graph
kNN_time = timeit.default_timer() - t1
print(f"RPT: metric={dist} n_trees={n_trees:2d} n_iters={n_iters:2d} leafSize={leafSize:2d} time={kNN_time:.4f}s")

# Leiden
K = 8
t1 = timeit.default_timer()
weighted_graph = utils.fast_weighted_sym_knng_igraph(indices[:, 1 : K], distances[:, 1 : K], use_exp_weight=False,verbose=False)
print('Graph Construction Time: {}'.format(timeit.default_timer() - t1))
t1 = timeit.default_timer()
labels = utils.run_leiden(weighted_graph)
print('Leiden Time: {}'.format(timeit.default_timer() - t1))
acc = utils.getMetric(labels, y)
print(f"#clusters: {int(acc[0])}, NMI: {acc[1]:.4f}, AMI: {acc[2]:.4f}, ARI: {acc[3]:.4f}")

# DANE
model = cluprop.cluprop()
model.n_threads = n_threads
K = 12 # K < k_max
t1 = timeit.default_timer()
model.knn_dane(indices[:, 1 : K], distances[:, 1 : K], K)
print('Dane Time: {}'.format(timeit.default_timer() - t1))
acc = utils.getMetric(np.array(model.labels_), y)
print(f"#clusters: {int(acc[0])}, NMI: {acc[1]:.4f}, AMI: {acc[2]:.4f}, ARI: {acc[3]:.4f}")