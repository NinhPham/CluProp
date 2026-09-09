#include "csr_graph.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

// Comparing two edges
bool by_neighbor_then_distance(const CSRGraph::Edge& lhs, const CSRGraph::Edge& rhs)
{
    if (lhs.neighbor != rhs.neighbor) {
        return lhs.neighbor < rhs.neighbor;
    }
    return lhs.distance < rhs.distance;
}

// Comparing two edges
bool by_distance_then_neighbor(const CSRGraph::Edge& lhs, const CSRGraph::Edge& rhs)
{
    if (lhs.distance != rhs.distance) {
        return lhs.distance < rhs.distance;
    }
    return lhs.neighbor < rhs.neighbor;
}

} // namespace

void CSRGraph::build_symmetric(
    const std::size_t n_vertices,
    const int* directed_neighbors,
    const float* directed_distances,
    const std::size_t n_neighbors, // k
    const int n_threads)
{
    if (n_threads <= 0) {
        throw std::invalid_argument("n_threads must be positive");
    }

    if (n_neighbors != 0 && n_vertices != 0
        && (!directed_neighbors || !directed_distances)) {
        throw std::invalid_argument("directed graph buffers must not be null");
    }

    std::vector<std::uint64_t> raw_degrees(n_vertices, 0);

#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int source = 0; source < n_vertices; ++source) {
        const std::size_t row_begin =
            static_cast<std::size_t>(source) * n_neighbors;
        for (std::size_t column = 0; column < n_neighbors; ++column) {
            const int target = directed_neighbors[row_begin + column];
            if (target == source) {
                continue;
            }

#pragma omp atomic update
            ++raw_degrees[static_cast<std::size_t>(source)];
#pragma omp atomic update
            ++raw_degrees[static_cast<std::size_t>(target)];
        }
    }

    std::vector<std::uint64_t> raw_offsets(n_vertices + 1, 0);
    for (std::size_t vertex = 0; vertex < n_vertices; ++vertex) {
        const std::uint64_t degree = raw_degrees[vertex];
        if (degree > std::numeric_limits<std::uint64_t>::max() - raw_offsets[vertex]) {
            throw std::length_error("CSR edge count exceeds uint64_t capacity");
        }
        raw_offsets[vertex + 1] = raw_offsets[vertex] + degree;
    }

    if (raw_offsets.back() > std::numeric_limits<std::size_t>::max()) {
        throw std::length_error("CSR edge count exceeds addressable memory");
    }

    std::vector<Edge> raw_edges(static_cast<std::size_t>(raw_offsets.back()));
    std::vector<std::uint64_t> fill_positions = raw_offsets;
    fill_positions.pop_back();

#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int source = 0; source < n_vertices; ++source) {
        const std::size_t row_begin =
            static_cast<std::size_t>(source) * n_neighbors;
        for (std::size_t column = 0; column < n_neighbors; ++column) {
            const int target = directed_neighbors[row_begin + column];
            if (target == source) {
                continue;
            }

            const float distance = directed_distances[row_begin + column];
            std::uint64_t forward_position;
            std::uint64_t reverse_position;

#pragma omp atomic capture
            forward_position = fill_positions[static_cast<std::size_t>(source)]++;
#pragma omp atomic capture
            reverse_position = fill_positions[static_cast<std::size_t>(target)]++;

            raw_edges[static_cast<std::size_t>(forward_position)] = {target, distance};
            raw_edges[static_cast<std::size_t>(reverse_position)] = {source, distance};
        }
    }

    std::vector<std::uint64_t> unique_degrees(n_vertices, 0);

#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int vertex = 0; vertex < n_vertices; ++vertex) {
        const std::size_t vertex_index = static_cast<std::size_t>(vertex);
        const std::size_t row_begin = static_cast<std::size_t>(raw_offsets[vertex_index]);
        const std::size_t row_end = static_cast<std::size_t>(raw_offsets[vertex_index + 1]);
        auto begin = raw_edges.begin() + row_begin;
        auto end = raw_edges.begin() + row_end;

        std::sort(begin, end, by_neighbor_then_distance);

        auto output = begin;
        for (auto input = begin; input != end; ++input) {
            if (output == begin || input->neighbor != (output - 1)->neighbor) {
                *output++ = *input;
            } else {
                (output - 1)->distance = std::min((output - 1)->distance, input->distance);
            }
        }

        std::sort(begin, output, by_distance_then_neighbor);
        unique_degrees[vertex_index] = static_cast<std::uint64_t>(output - begin);
    }

    std::vector<std::uint64_t> final_offsets(n_vertices + 1, 0);
    for (std::size_t vertex = 0; vertex < n_vertices; ++vertex) {
        final_offsets[vertex + 1] = final_offsets[vertex] + unique_degrees[vertex];
    }

    std::vector<Edge> final_edges(static_cast<std::size_t>(final_offsets.back()));

#pragma omp parallel for num_threads(n_threads) schedule(static)
    for (int vertex = 0; vertex < n_vertices; ++vertex) {
        const std::size_t vertex_index = static_cast<std::size_t>(vertex);
        const auto source = raw_edges.begin()
            + static_cast<std::size_t>(raw_offsets[vertex_index]);
        const auto destination = final_edges.begin()
            + static_cast<std::size_t>(final_offsets[vertex_index]);
        std::copy_n(
            source,
            static_cast<std::size_t>(unique_degrees[vertex_index]),
            destination);
    }

    n_vertices_ = n_vertices;
    offsets_ = std::move(final_offsets);
    edges_ = std::move(final_edges);
}

void CSRGraph::clear() noexcept
{
    n_vertices_ = 0;
    offsets_.clear();
    edges_.clear();
}

CSRGraph::NeighborView CSRGraph::neighbors(const int vertex) const
{
    if (vertex < 0 || vertex >= n_vertices_) {
        throw std::out_of_range("CSR vertex is out of range");
    }

    const std::size_t vertex_index = static_cast<std::size_t>(vertex);
    const std::size_t begin_offset = static_cast<std::size_t>(offsets_[vertex_index]);
    const std::size_t row_size = static_cast<std::size_t>(
        offsets_[vertex_index + 1] - offsets_[vertex_index]);
    static constexpr Edge empty_graph_sentinel{0, 0.0f};
    const Edge* row_data = edges_.empty()
        ? &empty_graph_sentinel
        : edges_.data() + begin_offset;
    return NeighborView(row_data, row_size);
}

std::size_t CSRGraph::degree(const int vertex) const
{
    return neighbors(vertex).size();
}

float CSRGraph::knn_distance(const int vertex, const std::size_t k) const
{
    const NeighborView row = neighbors(vertex);
    if (k == 0 || k > row.size()) {
        throw std::out_of_range("k exceeds the CSR vertex degree");
    }
    return row[k - 1].distance;
}
