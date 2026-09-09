#include "cluprop.h"
#include "utilities.h"
#include <queue>
#include <algorithm>
#include <iterator>
#include <fstream>
#include <cmath>


void cluprop::dane_simplified_(const int k)
{
    labels = IVector(n_points, -1);

    // boost::dynamic_bitset<> processSet(n_points);
    std::vector<uint8_t> vec_processed(n_points, 0);
    FVector minConnectedDist(n_points, POS_INF); // assign best_so_far distance
    FVector vec_kNNDist(n_points, POS_INF); // We indeed consider the point itself is its kNN, so we use (k-1)NN distance
    IVector vec_degree(n_points, 0); // We indeed consider the point itself is its kNN, so we use (k-1)NN distance

    IVector vec_density(n_points, 0);
    IVector sortedIndex_density = IVector(n_points, -1);

    // Note: if using avg kNN dist, then it might be useful with omp parallel for

#pragma omp parallel for num_threads(n_threads)
    for (int n = 0; n < n_points; ++n)
    {
        // init index from 0 to n
        sortedIndex_density[n] = n;
        const auto& Xi_neighbor = vec2D_NeighborDist_[n];

        // We can use kNN_dist as density proxy, however there are some case that points do not have enough kNN distance
        // We therefore use the degree as density proxy
        int Xi_degree = static_cast<int>(Xi_neighbor.size());

        vec_degree[n] = Xi_degree;
        vec_density[n] = Xi_degree;

        if (Xi_degree >= k)
            vec_kNNDist[n] = Xi_neighbor[k - 1].second;
    }

    // Sort points based on its density
    sort(sortedIndex_density.begin(), sortedIndex_density.end(),
        [&](int i1, int i2){
            return vec_density[i1] > vec_density[i2];
        }
    );

    // Note: We should use density to compute the average density for cluster
    // Note: Since distance range is too large, compare to 1/dist \in [0, 1]
    // Note: The sensitivity of the cluster quality is much better with density, compared to distance

    // Store cluster size
    vector<int> vecClusterSize;

    // Starting with cluster Id = -1
    int clusterId = -1;

    // Start from the highest density point idx
    for (const auto& topDens_Idx : sortedIndex_density)
    {
        // If it is already processed, then skip and go to next point
        if (vec_processed[topDens_Idx])
            continue;

        vec_processed[topDens_Idx] = true;

        // increase cluster Id
        clusterId = clusterId + 1;
        labels[topDens_Idx] = clusterId;
        vecClusterSize.emplace_back(1); // vecClusters contains cluster size for each cluster id

        // Min PQ has 3 values: (1) Xi, (2) Predecessor Idx, (3) weight
        Min_PQ_Triple seedSet;

        const auto& Xi_neighborhood = vec2D_NeighborDist_[topDens_Idx];

        // Insert all neighbor of Xi into PQ
        for (auto it = Xi_neighborhood.begin(); it != Xi_neighborhood.end(); ++it)
        {
            const auto& point = *it;

            int Xj = point.first; // first: idx, second: dist

            // only update if it is not processed
            if (vec_processed[Xj])
                continue;

            // Simulate Density-Peak, keep min connected distance with higher density points
            // This will reduce the size of PQ, improving running time
            if (minConnectedDist[Xj] < point.second) // point.second= dist(Xi, Xj)
                continue;

            // Heuristic to reduce PQ size: only add to PQ for smaller connected dist(Xi, Xj)
            // This idea is similar to Optics, i.e. keeping the minimum reachability dist so far
            minConnectedDist[Xj] = point.second;

            // There are some points which do not have enough k neighbors (in a general graph or mistakes on approximating kNN).
            // If so, we use d(Xi, Xj) as weight
            // This will help such border/noise points to be absorbed by the cluster formed by processed core points
            float weight = 0.0;
            if ((int)vec_degree[Xj] < k)
                weight = point.second;  // point does not have enough kNN
            else
                weight = (point.second + vec_kNNDist[Xj]) / 2; // point has enough kNN

            // Sorted by weight, but store extra information, i.e. highest-index = connected core point,
            // to form cluster
            seedSet.emplace(Xj, topDens_Idx, weight); // point idx, predecessor idx, weight

        }

        // Processing PQ for label propagation
        while (!seedSet.empty())
        {
            auto curTop = seedSet.top();
            seedSet.pop();

            int Xj = curTop.m_iIndex; // consider the new point which is connected by the highest density point

            if (vec_processed[Xj])
                continue;

            vec_processed[Xj] = true; // set processed
            int Xi = curTop.m_iPred; // get predecessor if Xj

            // Compute d(Xi, Xj) to decide whether to add Xj to the cluster of Xi (see the condition clusterSize > sVDC::min_cluster_size)
            float distXiXj = 0.0;
            if ((int)vec_degree[Xj] < k)
                distXiXj = curTop.m_fValue; // exact distXiXj from the weight
            else
                distXiXj = curTop.m_fValue * 2 - vec_kNNDist[Xj]; // dist(Xi, Xj) = (weight * 2 - kNN(Xj))

            int predLabel = labels[Xi];

            // if (predLabel < 0 || predLabel >= clusterId + 1)
            // {
            //     cout << "Bug in predLabel: " << predLabel << endl;
            //     continue;
            // }

            bool bExpandCluster = true;

            // If clusterSize < 50, then always propagate labels to its neighbors
            // Note: This is important to control the local expansion, e.g. not spreading too far away points
            if (vecClusterSize[predLabel] > min_cluster_size)
            {
                // Note: We should remove beta as we prefer less parameter to tune
                size_t t1 = min(k, vec_degree[Xi]);
                size_t t2 = min(k, vec_degree[Xj]);

                if (t1 == 0 || t2 == 0)
                    bExpandCluster = false;

                // If Xi and Xj are too far away, then we do not expand the cluster
                // This is to control the noise of approx neighborhoods returned by ANNS solvers
                // If Xj belongs to Xi's cluster, it should be connected via another point Xk, i.e.
                if ( distXiXj > (vec2D_NeighborDist_[Xi][t1 - 1].second + vec2D_NeighborDist_[Xj][t2 - 1].second))
                    bExpandCluster = false;
            }

            // If dist(Xi, Xj) is too large, then we halt expansion.
            // This happens at two border points
            if (!bExpandCluster) {
                vec_processed[Xj] = false;
                continue;
            }

            // We are now expand the cluster from Xj
            const auto& Xj_neighborhood = vec2D_NeighborDist_[Xj];

            // Note: Check one of kNN neighbors has label as the predecessor
            // as we want to spread cluster info via min reachability-dist
            bool hasPredLabel = false;

            for (auto it = Xj_neighborhood.begin(); it != Xj_neighborhood.begin() + min(k, vec_degree[Xj]); ++it)
            {
                if (labels[it->first] == predLabel)
                {
                    hasPredLabel = true;
                    break;
                }
            }

            // All kNN points do not have predecessor label, create new cluster
            if ( !hasPredLabel )
            {
                clusterId = clusterId + 1;
                labels[Xj] = clusterId;
                vecClusterSize.emplace_back(1);
            }
            else // Use the predecessor's label
            {
                labels[Xj] = predLabel; // label of predecessor
                vecClusterSize[predLabel] += 1;
            }

            // Now we extend the seedSet with Xj_neighborhood
            // Case 1: If Xj starts the new cluster, we tend to process the points around Xj in the new cluster
            // It this is the case, then we might process border points from previous cluster
            // This is why we keep predecessors' label to connect border points to previous cluster.
            // Case 2: If Xj is connected to the old cluster, we also extend PQ with Xj's neighbors
            for (auto it = Xj_neighborhood.begin(); it != Xj_neighborhood.end();++it)
            {
                const auto& p = *it;

                int Xk = p.first; // first: point idx, second: dist

                // only update if it is not processed
                if (vec_processed[Xk])
                    continue;

                // Note: This condition is nice to reduce PQ since we aim at finding min reachability distance
                if (minConnectedDist[Xk] < p.second)
                    continue;

                // Heuristic to reduce PQ size: only add to PQ for smaller connected dist(Xi, Xj)
                minConnectedDist[Xk] = p.second;

                float weight = 0.0;
                if (vec_degree[Xk] < k)
                    weight = p.second;
                else
                    weight = (p.second + vec_kNNDist[Xk]) / 2;

                seedSet.emplace(Xk, Xj, weight);

            }
        }
    }
}

/**
 * Execute density-aware neighborhood propagation
 * We form cluster by propagating labels from the highest density point to its neighbors
 * The priority queue is sorted by the distance to the connected higher density point and the kNN-distance of the new point
 *
 * Novelty:
 * - This is similar to Optics, but we use the highest density point to form initial seeds
 * - We keep track minConnectedDist to store the best-so-far distance between a point to the higher density point,
 * this will reduce the size of priority queue and ensure points are connected by the shortest distance to the higher density point
 * - By storing the predecessor, we are able to expand the cluster to all directions (like wavefront propagation) from highest density point to all lower-density points
 * and hence local support will play the key role on the edge/border points when ALL points from the cluster has been labeled.
 *
 * @param k: govern the density estimation
 * @param k_expand: govern the local support, i.e. checking k_expand nearest neighbors that has label of the predecessor
 *
 */
void cluprop::dane_(const int k, const int k_expand)
{
    if (verbose) {

        float avgSize = 0.0;

        // Counting points with empty neighborhoods, less than minPts, less than c*minPts
        int counter0 = 0, counter1 = 0, counter2 = 0;

        for (int n = 0; n < n_points; ++n)
        {
            auto const neighborSize = static_cast<float>(vec2D_NeighborDist_[n].size());

            if (neighborSize <= 0) {
                counter0++;
            }
            if (neighborSize < k) {
                counter1++;
            }
            if (neighborSize < k_expand) {
                counter2++;
            }

            avgSize += neighborSize;
        }

        avgSize /= n_points;

        cout << "Avg size = " << avgSize << endl;
        cout << "Number of points with empty neighborhoods: " << counter0 << endl;
        cout << "Number of points with less than " << k << " neighbors: " << counter1 << endl;
        cout << "Number of points with less than " << k_expand << " neighbors: " << counter2 << endl;
    }

    labels = IVector(n_points, -1);

    // boost::dynamic_bitset<> processSet(n_points);
    std::vector<uint8_t> vec_processed(n_points, 0);
    FVector minConnectedDist(n_points, POS_INF); // assign best_so_far distance
    FVector vec_kNNDist(n_points, POS_INF); // We indeed consider the point itself is its kNN, so we use (k-1)NN distance
    IVector vec_degree(n_points, 0); // We indeed consider the point itself is its kNN, so we use (k-1)NN distance
    IVector vec_expandLimit(n_points, 0);

    FVector vec_density(n_points, 0.0);
    IVector sortedIndex_density = IVector(n_points, -1);

    // Note: if using avg kNN dist, then it might be useful with omp parallel for

#pragma omp parallel for num_threads(n_threads)
    for (int n = 0; n < n_points; ++n)
    {
        // init index from 0 to n
        sortedIndex_density[n] = n;
        const auto& Xi_neighbor = vec2D_NeighborDist_[n];

        // This is for the case that some points do not have enough minPts neighbors
        // In this case, we use the size of neighborhood as density estimate
        // This might be true since points in the dense region should share similar closest random vectors
        // And we want to start the cluster from dense regions
        int Xi_degree = static_cast<float>(Xi_neighbor.size());

        // We heuristically use degree as density proxy though other kNN-Dist or avg-kNN-dist can be used.
        vec_degree[n] = Xi_degree;
        vec_density[n] = (float)Xi_degree;

        // We can use kNN dist. If not enough k neighbors, density = 0 (default)
        // if ( (int)vec2D_NeighborDist[n].size() >= k )
        // {
        //     float density_dist = vec2D_NeighborDist[n][k - 1].second; // minPts-1, since index starts from 0
        //
        //     if (density_dist > 0.0) // we might use [k - 1]
        //         vec_density[n] = 1.0 / density_dist;
        //     else
        //         vec_density[n] = 1.0 / EPSILON; // avoid division by zero
        // }

        // avg kNN-dist
        // for (int i = 0; k < k; ++k)
        // {
        //     float dist = vec2D_NeighborDist[n][k].second; // second: distance
        //     if (dist > 0) // we might use [k - 1]
        //         vec_density[n] += (dist / k);
        // }
        // vec_density[n] = vec_density[n] != 0.0 ? 1.0 / vec_density[n] : 0.0; // avoid division by zero

        vec_expandLimit[n] = Xi_degree; // default expanding to all neighbors
        if (propagation_cutoff)
            vec_expandLimit[n] = min(vec_expandLimit[n], k_expand);

        if (Xi_degree >= k)
            vec_kNNDist[n] = Xi_neighbor[k - 1].second;
    }

    // Sort points based on its density
    sort(sortedIndex_density.begin(), sortedIndex_density.end(),
        [&](int i1, int i2){
            return vec_density[i1] > vec_density[i2];
        }
    );

    // Note: We should use density to compute the average density for cluster
    // Note: Since distance range is too large, compare to 1/dist \in [0, 1]
    // Note: The sensitivity of the cluster quality is much better with density, compared to distance

    // Store cluster size
    vector<int> vecClusterSize;

    // Starting with cluster Id = -1
    int clusterId = -1;

    // Start from the highest density point idx
    for (const auto& topDens_Idx : sortedIndex_density)
    {
        // If it is already processed, then skip and go to next point
        if (vec_processed[topDens_Idx])
            continue;

        vec_processed[topDens_Idx] = true;

        // increase cluster Id
        clusterId = clusterId + 1;
        labels[topDens_Idx] = clusterId;
        vecClusterSize.emplace_back(1); // vecClusters contains cluster size for each cluster id

        // Min PQ has 3 values: (1) Xi, (2) Predecessor Idx, (3) weight
        Min_PQ_Triple seedSet;

        const auto& Xi_neighborhood = vec2D_NeighborDist_[topDens_Idx];

        // Insert all neighbor of Xi into PQ
        for (auto it = Xi_neighborhood.begin(); it != Xi_neighborhood.begin() + vec_expandLimit[topDens_Idx]; ++it)
        {
            const auto& point = *it;

            const int Xj = point.first; // first: idx, second: dist
            const float distXiXj = point.second;

            // if (Xj < 0 || Xj >= n_points)
            // {
            //     cout << "Bug in Xj: " << Xj << endl;
            //     continue;
            // }

            // only update if it is not processed
            if (vec_processed[Xj])
                continue;

            // Note: We might want to add more parameters to control the running time
            // This is for the case (2km + additional points) neighbors are too large and cover points are not on similar density, i.e. dist(Xi, Xj) >> kNN(Xi)
            // We will pick the first top-minPts points, then the rest depends on dist(Xi, Xj) < (1 +- alpha) kNN(Xi)
            // Since Xi_neighbor is sorted, so we should break
            // if (vec_density[topDens_Idx] * point.second > 1 + sVDC::alpha)
            //     break;

            // Since we process point using priority dist(p, x) + kNN_dist(x)
            // There is no need to push into the queue if dist(y, x) > dist(p, x)
            // minConnectedDist keep tracks the smallest distance from the predecessor to x
            // This idea is similar to Optics, i.e. keeping the minimum reachability dist so far
            if (minConnectedDist[Xj] < distXiXj)
                continue;

            minConnectedDist[Xj] = distXiXj;

            // There are some points which do not have enough k neighbors (in a general graph or mistakes on approximating kNN).
            // If so, we use d(Xi, Xj) as weight
            // This will help such border/noise points to be absorbed by the cluster formed by processed core points
            float weight = 0.0;
            if ((int)vec_degree[Xj] < k)
                weight = distXiXj;
            else
                weight = (distXiXj + vec_kNNDist[Xj]) / 2;

            // Sorted by weight, but store predecessor ID, to form cluster
            seedSet.emplace(Xj, topDens_Idx, weight); // point idx, predecessor idx, weight

        }

        // Processing PQ for label propagation
        while (!seedSet.empty())
        {
            auto curTop = seedSet.top();
            seedSet.pop();

            int Xj = curTop.m_iIndex; // consider the new point which is connected by a higher-density predecessor

            if (vec_processed[Xj])
                continue;

            vec_processed[Xj] = true; // set processed

            int Xi = curTop.m_iPred; // get predecessor

            // Recompute dist(Xi, Xj) as we only store weight = (d(Xi, Xj) + d_k(Xj)) / 2 and there is the case the point Xj does not have enough kNN
            float distXiXj = 0.0;
            if ((int)vec_degree[Xj] < k)
                distXiXj = curTop.m_fValue; // exact distXiXj from the weight
            else
                distXiXj = curTop.m_fValue * 2 - vec_kNNDist[Xj]; // dist(Xi, Xj) = (weight * 2 - kNN(Xj))

            int predLabel = labels[Xi];

            // if (predLabel < 0 || predLabel >= clusterId + 1)
            // {
            //     cout << "Bug in predLabel: " << predLabel << endl;
            //     continue;
            // }

            bool bExpandCluster = true;

            // If clusterSize < 50, then always propagate labels to its neighbors
            // Note: This is important to control the local expansion, e.g. not spreading too far away points
            if (vecClusterSize[predLabel] >= min_cluster_size)
            {
                // Note: We should remove beta as we prefer less parameter to tune
                size_t t1 = min(k, vec_degree[Xi]);
                size_t t2 = min(k, vec_degree[Xj]);

                if (t1 == 0 || t2 == 0)
                    bExpandCluster = false;

                // If Xi and Xj are too far away, then we do not expand the cluster
                // This is to control the noise of approx neighborhoods returned by ANNS solvers
                // If Xj belongs to Xi's cluster, it should be connected via another point Xk, i.e.
                if ( distXiXj > (vec2D_NeighborDist_[Xi][t1 - 1].second + vec2D_NeighborDist_[Xj][t2 - 1].second))
                    bExpandCluster = false;
            }

            // If dist(Xi, Xj) is too large, then we halt expansion.
            // This happens at two border points
            if (!bExpandCluster) {
                vec_processed[Xj] = false;

                // This predecessor was not an admissible connection.
                // Allow Xj to be reconsidered through another predecessor.
                minConnectedDist[Xj] = POS_INF;

                continue;
            }

            // We are now expand the cluster from Xj
            const auto& Xj_neighborhood = vec2D_NeighborDist_[Xj];

            // Note: Check one of kNN neighbors has label as the predecessor
            // as we want to spread cluster info via min reachability-dist
            bool hasPredLabel = false;

            for (auto it = Xj_neighborhood.begin(); it != Xj_neighborhood.begin() + min(k_expand, vec_degree[Xj]); ++it)
            {
                if (labels[it->first] == predLabel)
                {
                    hasPredLabel = true;
                    break;
                }
            }

            // All kNN points do not have predecessor label, create new cluster
            if ( !hasPredLabel )
            {
                clusterId = clusterId + 1;
                labels[Xj] = clusterId;
                vecClusterSize.emplace_back(1);
            }
            else // Use the predecessor's label
            {
                labels[Xj] = predLabel; // label of predecessor
                vecClusterSize[predLabel] += 1;
            }

            // Now we extend the seedSet with Xj_neighborhood
            // Case 1: If Xj starts the new cluster, we tend to process the points around Xj in the new cluster
            // It this is the case, then we might process border points from previous cluster
            // This is why we keep predecessors' label to connect border points to previous cluster.
            // Case 2: If Xj is connected to the old cluster, we also extend PQ with Xj's neighbors
            for (auto it = Xj_neighborhood.begin(); it != Xj_neighborhood.begin() + vec_expandLimit[Xj];++it)
            {
                const auto& p = *it;

                const int Xk = p.first; // first: point idx, second: dist
                const float distXjXk = p.second;

                // only update if it is not processed
                if (vec_processed[Xk])
                    continue;

                // Note: This condition is used to reduce PQ since we aim at finding min reachability distance
                if (minConnectedDist[Xk] < distXjXk)
                    continue;

                // Heuristic to reduce PQ size: only add to PQ for smaller connected dist(Xi, Xj)
                minConnectedDist[Xk] = distXjXk;

                float weight = 0.0;
                if (vec_degree[Xk] < k)
                    weight = distXjXk;
                else
                    weight = (distXjXk + vec_kNNDist[Xk]) / 2;

                seedSet.emplace(Xk, Xj, weight);

            }
        }
    }
}

/**
 * Wrapper function to call DNP() with matrix form of indices and distances (constructed externally by ANNS solvers)
 *
 * Algorithm:
 * - We construct sym_kNNG and store it in vec2D_NeighborDist
 *
 * @param matIndices: RowMajor matrix of indices, each row is the kNN indices for a point
 * @param matDistances: RowMajor matrix of distances, each row is the kNN distances for a point
 * @param k: govern the density estimation
 * @param k_expand: govern the local support, i.e. checking k_expand nearest neighbors that has label of the predecessor
 *
 */
void cluprop::knn_dane(const Ref<const RowMajorMatrixXi> & matIndices, const Ref<const RowMajorMatrixXf> & matDistances, const int k, const int k_expand)
{

    // Ensure kNN indices and distances, and parameter values are correct
    if (matIndices.rows() != matDistances.rows() || matIndices.cols() != matDistances.cols()) {
        throw std::invalid_argument(
            "indices and distances must have identical shapes");
     }

    if (k <= 0) {
        throw std::invalid_argument("k must be positive");
    }

    if (k_expand != -1 && k_expand <= 0) {
        throw std::invalid_argument(
            "k_expand must be -1 (simplified mode) or a positive integer");
    }

    n_points = matIndices.rows();
    const int n_neighbors = matIndices.cols();

    // Must remain serial: exceptions must not escape an OpenMP region.
    for (int n = 0; n < n_points; ++n) {
        for (int i = 0; i < n_neighbors; ++i) {
            const int iPointIdx = matIndices(n, i);
            const float fDist = matDistances(n, i);

            if (iPointIdx < 0 || iPointIdx >= n_points) {
                throw std::invalid_argument(
                    "indices contains an out-of-range point index");
            }

            if (!std::isfinite(fDist) || fDist < 0.0f) {
                throw std::invalid_argument(
                    "distances must be finite and non-negative");
            }
        }
    }

    // Step 1: Form symmetric kNN graph
    vec2D_NeighborDist_ = vector< vector< pair<int, float> > > (n_points, vector< pair<int, float> >());

    // Note: If NUM_LOCKS is large, we might not have enough stack memory if using array
    // 16K locks is good for million-point data set though it is not good for small data sets.
    constexpr size_t MAX_LOCKS = 16384;
    const size_t num_locks = std::min(static_cast<size_t>(n_points),MAX_LOCKS);
    vector<omp_lock_t> locks(num_locks);

    // Initialize locks
    for (size_t i = 0; i < num_locks; i++) {
        omp_init_lock(&locks[i]);
    }

    #pragma omp parallel for num_threads(n_threads)
    for (int n = 0; n < n_points; n++ ) {
        for (int i = 0; i < n_neighbors; ++i) {

            const int iPointIdx = matIndices(n, i); // vecIndices[n][i];
            const float fDist = matDistances(n, i); // vecDistances[n][i];

            if (iPointIdx == n) {
                continue;  // Self edges are harmless but unnecessary.
            }


            omp_set_lock(&locks[n % num_locks]);
            vec2D_NeighborDist_[n].emplace_back(iPointIdx, fDist); // duplicate at most twice
            omp_unset_lock(&locks[n % num_locks]);

            omp_set_lock(&locks[iPointIdx % num_locks]);
            vec2D_NeighborDist_[iPointIdx].emplace_back(n, fDist); // so vector is much better than map()
            omp_unset_lock(&locks[iPointIdx % num_locks]);
        }
    }

    // Destroy locks
    for (size_t i = 0; i < num_locks; i++) {
        omp_destroy_lock(&locks[i]);
    }


    // First, sorting by ID to remove the duplicate ID, potential some error with same ID but difference distance (x1, 0.1), (x1, 0.2)
    // After removing duplicated ID, then sort by distance ans DANE accesses points in the distance order
    #pragma omp parallel for num_threads(n_threads)
    for (int n = 0; n < n_points; ++n) {

        auto& neighbors = vec2D_NeighborDist_[n];

        // First sort: place all entries for the same neighbor together.
        std::sort(neighbors.begin(), neighbors.end(),
            [](const auto& a, const auto& b) {
                if (a.first != b.first) {
                    return a.first < b.first;
                }
                return a.second < b.second;
            });

        // Keep exactly one edge per neighbor: the one with minimum distance.
        std::vector<std::pair<int, float>> dedup;
        dedup.reserve(neighbors.size());

        for (const auto& edge : neighbors) {
            if (dedup.empty() || edge.first != dedup.back().first) {
                dedup.push_back(edge);
            } else {
                dedup.back().second =
                    std::min(dedup.back().second, edge.second);
            }
        }

        // Second sort: DANE requires nearest neighbors first.
        std::sort(dedup.begin(), dedup.end(),
            [](const auto& a, const auto& b) {
                if (a.second != b.second) {
                    return a.second < b.second;
                }
                return a.first < b.first;
            });

        neighbors = std::move(dedup);
    }

    // Sorting vec2D_NeighborDist[n] by distance
    // #pragma omp parallel for num_threads(n_threads)
    // for (int n = 0; n < n_points; ++n) {
    //
    //     // Step 1: Sort by value (float)
    //     std::sort(vec2D_NeighborDist_[n].begin(), vec2D_NeighborDist_[n].end(), [](const auto& a, const auto& b) {
    //         // Compare based on the float value first
    //         if (a.second != b.second) {
    //             return a.second < b.second; // Sort by float in ascending order
    //         }
    //         // If float values are equal, compare based on the int value
    //         return a.first < b.first; // Sort by int in ascending order
    //     });
    //
    //
    //     // Step 2: Linear scan and merge duplicates
    //     std::vector<std::pair<int, float>> dedup;
    //     // dedup.reserve(sVDC::vec2D_NeighborDist[n].size());  // optional optimization
    //
    //     for (size_t i = 0; i < vec2D_NeighborDist_[n].size(); ++i) {
    //         if (dedup.empty() || vec2D_NeighborDist_[n][i].first != dedup.back().first) {
    //             dedup.push_back(vec2D_NeighborDist_[n][i]);
    //         } else {
    //             // Keep min distance value (can switch to max or average)
    //             dedup.back().second = min(dedup.back().second, vec2D_NeighborDist_[n][i].second);
    //         }
    //     }
    //
    //     vec2D_NeighborDist_[n] = dedup;
    // }

    if (verbose)
    {
        float avgSize = 0.0;
        int counter0 = 0, counter1 = 0;
        for (int n = 0; n < n_points; ++n) {
            if (vec2D_NeighborDist_[n].empty()) {
                counter0++;
            }
            if ((int)vec2D_NeighborDist_[n].size() < k) {
                counter1++;
            }
            avgSize += vec2D_NeighborDist_[n].size();
        }

        avgSize /= n_points;

        cout << "Avg size = " << avgSize << endl;
        cout << "Number of points with empty neighborhoods: " << counter0 << endl;
        cout << "Number of points with less than " << k << " neighbors: " << counter1 << endl;
    }


    // Step 2: Call propagation
    if (k_expand == -1)
        dane_simplified_(k);
        // prop_(k, "OPTICS");
    else
        dane_(k, k_expand);
}

void cluprop::prop_(const int k, const string reachDistType)
{
    labels = IVector(n_points, -1);

    // boost::dynamic_bitset<> processSet(n_points);
    std::vector<uint8_t> vec_processed(n_points, 0);
    FVector minConnectedDist(n_points, POS_INF); // assign best_so_far distance
    FVector vec_kNNDist(n_points, POS_INF); // We indeed consider the point itself is its kNN, so we use (k-1)NN distance
    IVector vec_degree(n_points, 0); // We indeed consider the point itself is its kNN, so we use (k-1)NN distance

    IVector vec_density(n_points, 0);
    IVector sortedIndex_density = IVector(n_points, -1);

    // Note: if using avg kNN dist, then it might be useful with omp parallel for

#pragma omp parallel for num_threads(n_threads)
    for (int n = 0; n < n_points; ++n)
    {
        // init index from 0 to n
        sortedIndex_density[n] = n;
        const auto& Xi_neighbor = vec2D_NeighborDist_[n];

        // This is for the case that some points do not have enough minPts neighbors
        // In this case, we use the size of neighborhood as density estimate
        // This might be true since points in the dense region should share similar closest random vectors
        // And we want to start the cluster from dense regions
        int Xi_degree = static_cast<int>(Xi_neighbor.size());

        vec_degree[n] = Xi_degree;
        vec_density[n] = Xi_degree;

        if (Xi_degree >= k)
            vec_kNNDist[n] = Xi_neighbor[k - 1].second;
    }

    // Sort points based on its density
    sort(sortedIndex_density.begin(), sortedIndex_density.end(),
        [&](int i1, int i2){
            return vec_density[i1] > vec_density[i2];
        }
    );

    // Note: We should use density to compute the average density for cluster
    // Note: Since distance range is too large, compare to 1/dist \in [0, 1]
    // Note: The sensitivity of the cluster quality is much better with density, compared to distance

    // Store cluster size
    vector<int> vecClusterSize;

    // Starting with cluster Id = -1
    int clusterId = -1;

    // Start from the highest density point idx
    for (const auto& Xi : sortedIndex_density)
    {
        // If it is already processed, then skip and go to next point
        if (vec_processed[Xi])
            continue;

        vec_processed[Xi] = true;

        // increase cluster Id
        clusterId = clusterId + 1;
        labels[Xi] = clusterId;
        vecClusterSize.emplace_back(1); // vecClusters contains cluster size for each cluster id

        // Min PQ has 3 values: (1) Xi, (2) Predecessor Idx, (3) weight
        Min_PQ_Triple seedSet;

        const auto& Xi_neighborhood = vec2D_NeighborDist_[Xi];

        // Insert all neighbor of Xi into PQ
        for (auto it = Xi_neighborhood.begin(); it != Xi_neighborhood.end(); ++it)
        {
            const auto& point = *it;

            int Xj = point.first; // first: idx, second: dist

            // only update if it is not processed
            if (vec_processed[Xj])
                continue;

            // Simulate Density-Peak, keep min connected distance with higher density points
            // This will reduce the size of PQ, improving running time
            if (minConnectedDist[Xj] < point.second) // point.second= dist(Xi, Xj)
                continue;

            // Heuristic to reduce PQ size: only add to PQ for smaller connected dist(Xi, Xj)
            // This idea is similar to Optics, i.e. keeping the minimum reachability dist so far
            minConnectedDist[Xj] = point.second;

            // There are some points which do not have enough k neighbors (in a general graph or mistakes on approximating kNN).
            // If so, we use d(Xi, Xj) as weight
            // This will help such border/noise points to be absorbed by the cluster formed by processed core points
            float weight = compute_reachability(vec_kNNDist[Xi], vec_kNNDist[Xj], point.second, reachDistType);

            // Sorted by weight, but store extra information, i.e. highest-index = connected core point,
            // to form cluster
            seedSet.emplace(Xj, Xi, weight); // point idx, predecessor idx, weight

        }

        // Processing PQ for label propagation
        while (!seedSet.empty())
        {
            auto curTop = seedSet.top();
            seedSet.pop();

            int Xj = curTop.m_iIndex; // consider the new point which is connected by the highest density point

            if (vec_processed[Xj])
                continue;

            vec_processed[Xj] = true; // set processed
            int Xi = curTop.m_iPred; // get predecessor if Xj

            int predLabel = labels[Xi];

            // We are now expand the cluster from Xj
            const auto& Xj_neighborhood = vec2D_NeighborDist_[Xj];

            // Note: Check one of kNN neighbors has label as the predecessor
            // as we want to spread cluster info via min reachability-dist
            bool hasPredLabel = false;

            for (auto it = Xj_neighborhood.begin(); it != Xj_neighborhood.begin() + min(k, vec_degree[Xj]); ++it)
            {
                if (labels[it->first] == predLabel)
                {
                    hasPredLabel = true;
                    break;
                }
            }

            // All kNN points do not have predecessor label, create new cluster
            if ( !hasPredLabel )
            {
                clusterId = clusterId + 1;
                labels[Xj] = clusterId;
                vecClusterSize.emplace_back(1);
            }
            else // Use the predecessor's label
            {
                labels[Xj] = predLabel; // label of predecessor
                vecClusterSize[predLabel] += 1;
            }

            // Now we extend the seedSet with Xj_neighborhood
            // Case 1: If Xj starts the new cluster, we tend to process the points around Xj in the new cluster
            // It this is the case, then we might process border points from previous cluster
            // This is why we keep predecessors' label to connect border points to previous cluster.
            // Case 2: If Xj is connected to the old cluster, we also extend PQ with Xj's neighbors
            for (auto it = Xj_neighborhood.begin(); it != Xj_neighborhood.end();++it)
            {
                const auto& p = *it;

                int Xk = p.first; // first: point idx, second: dist

                // only update if it is not processed
                if (vec_processed[Xk])
                    continue;

                // Note: This condition is nice to reduce PQ since we aim at finding min reachability distance
                if (minConnectedDist[Xk] < p.second)
                    continue;

                // Heuristic to reduce PQ size: only add to PQ for smaller connected dist(Xi, Xj)
                minConnectedDist[Xk] = p.second;

                float weight = compute_reachability(vec_kNNDist[Xj], vec_kNNDist[Xk], p.second, reachDistType);

                seedSet.emplace(Xk, Xj, weight);

            }
        }
    }
}










