import time
import numpy as np
from sklearn.cluster import DBSCAN

# 70,000 x 786 random matrix
n, d = 70000, 786
X = np.random.rand(n, d).astype(np.float32)

# DBSCAN with multi-threading
n_jobs=8
dbscan = DBSCAN(eps=3.0, min_samples=5, metric="euclidean", n_jobs=n_jobs)

start = time.perf_counter()
labels = dbscan.fit_predict(X)
end = time.perf_counter()

print(f"Runtime: {end - start:.4f} seconds")
print(f"Number of clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
print(f"Number of noise points: {(labels == -1).sum()}")
