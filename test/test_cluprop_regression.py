import numpy as np
import pytest

import cluprop


@pytest.mark.parametrize("n_threads", [1, 4])
def test_simplified_dane_labels_match_pre_csr_result(n_threads):
    indices = np.array([
        [1, 2],
        [0, 2],
        [0, 1],
        [4, 4],
        [3, 3],
    ], dtype=np.int32)
    distances = np.array([
        [1.0, 2.0],
        [1.0, 1.5],
        [2.0, 1.5],
        [1.0, 2.0],
        [1.0, 2.0],
    ], dtype=np.float32)

    model = cluprop.cluprop()
    model.n_threads = n_threads
    model.knn_dane(indices, distances, k=2)

    assert list(model.labels_) == [0, 0, 0, 1, 1]


@pytest.mark.parametrize("n_threads", [1, 4])
def test_local_support_dane_labels_match_pre_csr_result(n_threads):
    indices = np.array([[1], [2], [3], [3]], dtype=np.int32)
    distances = np.array([[1.0], [1.0], [1.0], [0.0]], dtype=np.float32)

    model = cluprop.cluprop()
    model.n_threads = n_threads
    model.knn_dane(indices, distances, k=1, k_expand=1)

    assert list(model.labels_) == [0, 0, 0, 0]
