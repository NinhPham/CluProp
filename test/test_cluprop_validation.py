import numpy as np
import pytest
import cluprop


@pytest.fixture
def valid_knn():
    indices = np.array([
        [1, 2],
        [0, 2],
        [1, 0],
    ], dtype=np.int32)

    distances = np.array([
        [1.0, 2.0],
        [1.0, 1.0],
        [1.0, 2.0],
    ], dtype=np.float32)

    return indices, distances


def test_rejects_mismatched_knn_shapes(valid_knn):
    indices, distances = valid_knn
    model = cluprop.cluprop(3, 1)

    with pytest.raises(ValueError, match="identical shapes"):
        model.knn_dane(indices, distances[:, :1], k=1)


def test_rejects_non_positive_k(valid_knn):
    indices, distances = valid_knn
    model = cluprop.cluprop(3, 1)

    with pytest.raises(ValueError, match="k must be positive"):
        model.knn_dane(indices, distances, k=0)


def test_rejects_invalid_k_expand(valid_knn):
    indices, distances = valid_knn
    model = cluprop.cluprop(3, 1)

    with pytest.raises(ValueError, match="k_expand"):
        model.knn_dane(indices, distances, k=1, k_expand=0)


def test_rejects_out_of_range_neighbor_index(valid_knn):
    indices, distances = valid_knn
    indices[0, 0] = 3
    model = cluprop.cluprop(3, 1)

    with pytest.raises(ValueError, match="out-of-range"):
        model.knn_dane(indices, distances, k=1)


def test_rejects_non_finite_or_negative_distance(valid_knn):
    indices, distances = valid_knn
    distances[0, 0] = np.nan
    model = cluprop.cluprop(3, 1)

    with pytest.raises(ValueError, match="finite and non-negative"):
        model.knn_dane(indices, distances, k=1)
