import numpy as np
import pandas as pd

# Example: Load your dataset here
path = "/shared/Dataset/Clustering/"

# y = np.loadtxt(path + 'mnist_all_y_70K_784')
# filename = "/shared/Dataset/Clustering/mnist_all_X"
#
y = np.loadtxt(path + 'mnist8m_y_8100000_784')
filename = "/shared/Dataset/Clustering/mnist8m_X"
X = np.loadtxt(filename)

n = np.shape(y)[0]

rows_to_delete = [i for i in range(n) if np.random.uniform() < 0.9]
new_X = np.delete(X, rows_to_delete, axis=0)
new_y = np.delete(y, rows_to_delete, axis=0)

print("Remaining samples:", new_X.shape[0])

# X_sampled and y_sampled now contain 90% of the data for each label
np.savetxt(path + 'mnist8m_X_10', new_X, fmt='%d')
np.savetxt(path + 'mnist8m_y_10', new_y, fmt='%d')
