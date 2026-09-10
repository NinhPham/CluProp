#include "csr_graph.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#define REQUIRE(condition) \
    do { \
        if (!(condition)) { \
            throw std::runtime_error("test requirement failed: " #condition); \
        } \
    } while (false)

namespace {

using LegacyGraph = std::vector<std::vector<std::pair<int, float>>>;

LegacyGraph build_legacy_graph(
    const int vertex_count,
    const std::vector<int>& indices,
    const std::vector<float>& distances,
    const std::size_t row_width)
{
    LegacyGraph graph(static_cast<std::size_t>(vertex_count));

    for (int source = 0; source < vertex_count; ++source) {
        const std::size_t row_begin = static_cast<std::size_t>(source) * row_width;
        for (std::size_t column = 0; column < row_width; ++column) {
            const int target = indices[row_begin + column];
            const float distance = distances[row_begin + column];
            if (target == source) {
                continue;
            }
            graph[static_cast<std::size_t>(source)].emplace_back(target, distance);
            graph[static_cast<std::size_t>(target)].emplace_back(source, distance);
        }
    }

    for (auto& row : graph) {
        std::sort(row.begin(), row.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.first != rhs.first) {
                return lhs.first < rhs.first;
            }
            return lhs.second < rhs.second;
        });

        std::vector<std::pair<int, float>> deduplicated;
        deduplicated.reserve(row.size());
        for (const auto& edge : row) {
            if (deduplicated.empty() || edge.first != deduplicated.back().first) {
                deduplicated.push_back(edge);
            } else {
                deduplicated.back().second =
                    std::min(deduplicated.back().second, edge.second);
            }
        }

        std::sort(
            deduplicated.begin(),
            deduplicated.end(),
            [](const auto& lhs, const auto& rhs) {
                if (lhs.second != rhs.second) {
                    return lhs.second < rhs.second;
                }
                return lhs.first < rhs.first;
            });
        row = std::move(deduplicated);
    }

    return graph;
}

void assert_matches_legacy(const CSRGraph& actual, const LegacyGraph& expected)
{
    REQUIRE(actual.vertex_count() == static_cast<int>(expected.size()));

    std::size_t expected_edge_count = 0;
    for (int vertex = 0; vertex < actual.vertex_count(); ++vertex) {
        const auto neighbors = actual.neighbors(vertex);
        const auto& expected_row = expected[static_cast<std::size_t>(vertex)];
        REQUIRE(actual.degree(vertex) == expected_row.size());
        REQUIRE(neighbors.size() == expected_row.size());

        std::size_t index = 0;
        for (const auto& edge : neighbors) {
            REQUIRE(edge.neighbor == expected_row[index].first);
            REQUIRE(edge.distance == expected_row[index].second);
            REQUIRE(neighbors[index].neighbor == edge.neighbor);
            REQUIRE(neighbors[index].distance == edge.distance);
            ++index;
        }
        expected_edge_count += expected_row.size();
    }

    REQUIRE(actual.edge_count() == expected_edge_count);
}

} // namespace

int main()
{
    constexpr int vertex_count = 7;
    constexpr std::size_t row_width = 4;

    // Includes asymmetric and reciprocal edges, duplicates with different
    // distances, self edges, and two isolated vertices.
    const std::vector<int> indices = {
        0, 1, 1, 2,
        2, 0, 1, 2,
        3, 2, 1, 3,
        2, 4, 3, 4,
        3, 4, 0, 0,
        5, 5, 5, 5,
        6, 6, 6, 6,
    };
    const std::vector<float> distances = {
        0.0f, 5.0f, 2.0f, 4.0f,
        3.0f, 6.0f, 0.0f, 1.0f,
        7.0f, 0.0f, 2.0f, 2.0f,
        8.0f, 1.0f, 0.0f, 3.0f,
        5.0f, 0.0f, 9.0f, 8.0f,
        0.0f, 1.0f, 2.0f, 3.0f,
        0.0f, 1.0f, 2.0f, 3.0f,
    };

    const LegacyGraph legacy =
        build_legacy_graph(vertex_count, indices, distances, row_width);
    const LegacyGraph expected = {
        {{1, 2.0f}, {2, 4.0f}, {4, 8.0f}},
        {{2, 1.0f}, {0, 2.0f}},
        {{1, 1.0f}, {3, 2.0f}, {0, 4.0f}},
        {{4, 1.0f}, {2, 2.0f}},
        {{3, 1.0f}, {0, 8.0f}},
        {},
        {},
    };
    REQUIRE(legacy == expected);

    CSRGraph serial_graph;
    serial_graph.build_symmetric(
        vertex_count, indices.data(), distances.data(), row_width, 1);
    assert_matches_legacy(serial_graph, legacy);

    CSRGraph parallel_graph;
    parallel_graph.build_symmetric(
        vertex_count, indices.data(), distances.data(), row_width, 4);
    assert_matches_legacy(parallel_graph, legacy);

    REQUIRE(serial_graph.knn_distance(0, 1) == 2.0f);
    REQUIRE(serial_graph.knn_distance(0, 2) == 4.0f);
    REQUIRE(serial_graph.knn_distance(0, 3) == 8.0f);
    REQUIRE(serial_graph.neighbors(5).empty());
    REQUIRE(serial_graph.neighbors(6).begin() == serial_graph.neighbors(6).end());

    return 0;
}
