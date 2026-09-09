#ifndef CSR_GRAPH_H
#define CSR_GRAPH_H

#include <cstddef>
#include <cstdint>
#include <vector>

class CSRGraph {
public:
    struct Edge {
        int neighbor;
        float distance;
    };

    class NeighborView {
    public:
        using const_iterator = const Edge*;

        NeighborView(const Edge* data, std::size_t size) noexcept
            : data_(data), size_(size) {}

        const_iterator begin() const noexcept { return data_; }
        const_iterator end() const noexcept {
            return size_ == 0 ? data_ : data_ + size_;
        }
        std::size_t size() const noexcept { return size_; }
        bool empty() const noexcept { return size_ == 0; }
        const Edge& operator[](std::size_t index) const noexcept {
            return data_[index];
        }

    private:
        const Edge* data_;
        std::size_t size_;
    };

    void build_symmetric(
        std::size_t n_vertices,
        const int* directed_neighbors,
        const float* directed_distances,
        std::size_t n_neighbors,
        int n_threads);

    void clear() noexcept;

    NeighborView neighbors(int vertex) const;
    std::size_t degree(int vertex) const;
    float knn_distance(int vertex, std::size_t k) const;

    int vertex_count() const noexcept { return n_vertices_; }
    std::size_t edge_count() const noexcept { return edges_.size(); }

private:
    int n_vertices_ = 0;
    std::vector<std::uint64_t> offsets_;
    std::vector<Edge> edges_;
};

#endif // CSR_GRAPH_H
