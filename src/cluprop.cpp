#include "cluprop.h"
#include "utilities.h"
#include <queue>
#include <algorithm>
#include <iterator>
#include <fstream>

/**
 *
 * @param: MATRIX_X : Eigen matrix X
 * @param knn_alg : Algorithm for approximate kNN
 * @param graph_type : Type of graph (default symmetric kNN)
 * @param propagation_alg : Propagation algorithm: DNP or DBSCAN
 * @param k
 * @param c
 */
// void cluprop::fit(const Ref<const RowMajorMatrixXf> & MATRIX_X, const string& knn_alg, int k, float c)
// {
//     // Step 1: Copy data, check support distance
//     if (verbose)
//     {
//         cout << "k: " << k << endl;
//
//         cout << "distance: " << distance << endl;
//         cout << "n_threads: " << n_threads << endl;
//     }
//
//     // omp_set_dynamic(0);     // Explicitly disable dynamic teams
//     omp_set_num_threads(n_threads);
//
//     chrono::steady_clock::time_point begin, local_begin;
//     begin = chrono::steady_clock::now();
//     matrix_X = MATRIX_X;
//     transformData(matrix_X, distance);
//
//     if (verbose)
//         cout << "Copy data and check supporting distance time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;
//
//     // Step 2: kNN graph construction
//     begin = chrono::steady_clock::now();
//     if (knn_alg == "brute") {
//         local_begin = chrono::steady_clock::now();
//         bf_sym_Gk_(k);
//
//         if (verbose)
//             cout << "Bruteforce kNN graph construction time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - local_begin).count() << "[ms]" << endl;
//
//     }
//
//
//     // Step 3: Propagation
//     begin = chrono::steady_clock::now();
//     dnp_(k, c);
//     if (verbose)
//         cout << "Run DNP time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;
// }

/**
 * Compute exact kNN using bruteforce method using distance matrix of cluprop
 *
 * @param dataset
 * @param k
 * @return: Row-major matrix of indices and distances (N x k)
 */
// tuple<MatrixXi, MatrixXf> cluprop::brute_knn(const string& dataset, const int k)
// {
//     if (verbose)
//     {
//         cout << "n_points: " << n_points << endl;
//         cout << "n_features: " << n_features << endl;
//
//         cout << "distance: " << distance << endl;
//         cout << "n_threads: " << n_threads << endl;
//     }
//
//     // omp_set_dynamic(0);     // Explicitly disable dynamic teams
//     omp_set_num_threads(n_threads);
//
//     chrono::steady_clock::time_point begin, start;
//
//     begin = chrono::steady_clock::now();
//
//     // loadtxtData(dataset, sVDC::distance, sVDC::n_points, sVDC::n_features, sVDC::matrix_X);
//     loadbinData(dataset, distance, n_points, n_features, matrix_X);
//
//     if (verbose)
//         cout << "Loading data time = " << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]" << endl;
//
//     begin = chrono::steady_clock::now();
//
//     MatrixXi matrix_indices_ = -Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Ones(n_points, k);
//     MatrixXf matrix_distances_ = -Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Ones(n_points, k);
//
//     // Simple parallel that executes n^2 distance computations
// #pragma omp parallel for
//     for (int n1 = 0; n1 < n_points; ++n1)
//     {
//         VectorXf vecXn = matrix_X.row(n1);
//
//         priority_queue <IFPair, vector<IFPair>> vectorMaxQue_TopK;
//
//         for (int n2 = 0; n2 < n_points; ++n2)
//         {
//             if (n2 == n1)
//                 continue;
//
//             float dist = computeDist(vecXn,  matrix_X.row(n2), distance);
//
//             if ((int)vectorMaxQue_TopK.size() < k)
//                 vectorMaxQue_TopK.emplace(n2, dist);
//
//             else if (dist < vectorMaxQue_TopK.top().m_fValue)
//             {
//                 vectorMaxQue_TopK.pop();
//                 vectorMaxQue_TopK.emplace(n2, dist);
//             }
//         }
//
//         int k_idx = k - 1;
//         while (!vectorMaxQue_TopK.empty())
//         {
//             IFPair pair = vectorMaxQue_TopK.top(); // pointIdx, dist
//             vectorMaxQue_TopK.pop();
//
//             matrix_indices_(n1, k_idx) = pair.m_iIndex;
//             matrix_distances_(n1, k_idx) = pair.m_fValue;
//
//             k_idx--;
//         }
//     }
//
//
//     if (verbose) {
//         cout << "Bruteforce computation time = "
//              << chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now() - begin).count() << "[ms]"
//              << endl;
//
//         // Write binary
//         string filename = distance + "_k_" + int2str(k) + "_indices.bin";
//         std::ofstream outIdx(filename, std::ios::binary);
//         outIdx.write(reinterpret_cast<const char*>(matrix_indices_.data()), matrix_indices_.size() * sizeof(int));
//         outIdx.close();
//
//         filename = distance + "_k_" + int2str(k) + "_distances.bin";
//         std::ofstream outDist(filename, std::ios::binary);
//         outDist.write(reinterpret_cast<const char*>(matrix_distances_.data()), matrix_distances_.size() * sizeof(float));
//         outDist.close();
//
//     }
//
//     return {matrix_indices_, matrix_distances_};
// }

/**
 * Execute density-aware neighborhood propagation
 * We form cluster by propagating labels from the highest density point to its neighbors
 * The priority queue is sorted by the distance to the connected higher density point and the kNN-distance of the new point
 *
 * Novelty:
 * - This is similar to Optics, but we use the highest density point to form initial seeds
 * - We keep track minConnectedDist to store the best-so-far distance between a point to the higher density point,
 * this will reduce the size of priority queue and ensure points are connected by the shortest distance to the higher density point
 *
 *
 * @param k: govern the density estimation
 * @param k_expand: govern the size of neighborhood to check the label consistency to decide whether to add the new point to the cluster
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

    boost::dynamic_bitset<> processSet(n_points);
    FVector minConnectedDist(n_points, POS_INF); // assign best_so_far distance

    FVector vec_density(n_points, 0.0);
    IVector sortedIndex_density = IVector(n_points, -1);

    // Note: if using avg kNN dist, then it might be useful with omp parallel for
// #pragma omp parallel for
    for (int n = 0; n < n_points; ++n)
    {
        // init index from 0 to n
        sortedIndex_density[n] = n;

        // This is for the case that some points do not have enough minPts neighbors
        // In this case, we use the size of neighborhood as density estimate
        // This might be true since points in the dense region should share similar closest random vectors
        // And we want to start the cluster from dense regions
        vec_density[n] = static_cast<float>(vec2D_NeighborDist_[n].size());

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
    }

    sort(sortedIndex_density.begin(), sortedIndex_density.end(),
        [&](int i1, int i2){
            return vec_density[i1] > vec_density[i2];
        }
    );

    // Note: We still need to use density to compute the average density for cluster
    // Note: Since distance range is too large, compare to 1/dist \in [0, 1]
    // Note: The sensitivity of the cluster quality is much better with density, compared to distance

    // Store cluster size
    vector<int> vecClusters;

    // Starting with cluster Id = -1
    int clusterId = -1;

    // Start from the highest density point idx
    for (const auto& topDens_Idx : sortedIndex_density)
    {
        // If it is already processed, then skip and go to next point
        if (processSet[topDens_Idx])
            continue;

        processSet[topDens_Idx] = true;

        // increase cluster Id
        clusterId = clusterId + 1;
        labels[topDens_Idx] = clusterId;
        vecClusters.emplace_back(1); // vecClusters contains cluster size for each cluster id

        // Min PQ has 3 values: (1) Xi, (2) Predecessor Idx, (3) weight
        Min_PQ_Triple seedSet;

        const auto& Xi_neighborhood = vec2D_NeighborDist_[topDens_Idx];

        // For all Xj is neighbor of core Xi, insert into the PQ with its predecessor Xi
        // We use sVDC::neighbor_cutoff to control the size of neighborhood to insert into PQ
        int Xi_neighborSize = static_cast<int>(Xi_neighborhood.size());
        if (propagation_cutoff)
            Xi_neighborSize = min(Xi_neighborSize, k_expand);

//        for (const auto & point : Xi_neighborhood)
        for (auto it = Xi_neighborhood.begin();
                 it != Xi_neighborhood.begin() + Xi_neighborSize;
                 ++it)
        {
            const auto& point = *it;

            int Xj = point.first; // first: idx, second: dist

            // if (Xj < 0 || Xj >= n_points)
            // {
            //     cout << "Bug in Xj: " << Xj << endl;
            //     continue;
            // }

            // only update if it is not processed
            if (processSet[Xj])
                continue;

            // Note: We might want to add more parameters to control the running time
            // This is for the case (2km + additional points) neighbors are too large and cover points are not on similar density, i.e. dist(Xi, Xj) >> kNN(Xi)
            // We will pick the first top-minPts points, then the rest depends on dist(Xi, Xj) < (1 +- alpha) kNN(Xi)
            // Since Xi_neighbor is sorted, so we should break
            // if (vec_density[topDens_Idx] * point.second > 1 + sVDC::alpha)
            //     break;

            // Simulate Density-Peak, keep min connected distance with higher density points
            // This will reduce the size of PQ, improving running time
            if (minConnectedDist[Xj] < point.second) // point.second= dist(Xi, Xj)
                continue;

            // Heuristic to reduce PQ size: only add to PQ for smaller connected dist(Xi, Xj)
            // This idea is similar to Optics, i.e. keeping the minimum reachability dist so far
            minConnectedDist[Xj] = point.second;

            // Xi_neighborhood[j].second = dist(Xi, Xj)
            // float weight = (Xi_neighborhood[j].second + sOptics::vec_CoreDist[Xj]) / 2;
            // There are border/noise points which do not have enough k neighbors. If so, we use d(Xi, Xj) as weight
            // This will help such border/noise points to be absorbed by the cluster formed by processed core points
            float weight = 0.0;
            if ((int)vec2D_NeighborDist_[Xj].size() < k)
                weight = point.second;
            else
                weight = (point.second + vec2D_NeighborDist_[Xj][k - 1].second) / 2;

            // Sorted by weight, but store extra information, i.e. highest-index = connected core point,
            // to form cluster
            seedSet.emplace(Xj, topDens_Idx, weight); // point idx, predecessor idx, weight

        }

        // Processing PQ for label propagation
        while (!seedSet.empty())
        {
            int Xj = seedSet.top().m_iIndex; // consider the new point which is connected by the highest density point
            int Xi = seedSet.top().m_iPred;

            // Compute d(Xi, Xj) to decide whether to add Xj to the cluster of Xi (see the condition clusterSize > sVDC::min_cluster_size)
            float distXiXj = 0.0;
            if ((int)vec2D_NeighborDist_[Xj].size() < k)
                distXiXj = seedSet.top().m_fValue;
            else
                distXiXj = seedSet.top().m_fValue * 2 - vec2D_NeighborDist_[Xj][k - 1].second; // dist(Xi, Xj) = (weight * 2 - kNN(Xj))

            seedSet.pop();

            if (processSet[Xj])
                continue;

            processSet[Xj] = true; // set processed

            int predLabel = labels[Xi];

            // if (predLabel < 0 || predLabel >= clusterId + 1)
            // {
            //     cout << "Bug in predLabel: " << predLabel << endl;
            //     continue;
            // }

            int clusterSize = vecClusters[predLabel];

            bool bExpandCluster = true;

            // If clusterSize < 50, then always propagate labels to its neighbors
            // Note: This is important to control the local expansion, e.g. not spreading too far away points
            if (clusterSize > min_cluster_size)
            {
                // Note: We should remove beta as we prefer less parameter to tune
                size_t t1 = min((size_t)k, vec2D_NeighborDist_[Xi].size());
                size_t t2 = min((size_t)k, vec2D_NeighborDist_[Xj].size());

                // if (t1 == 0 || t2 == 0)
                //     cout << "Bug in distXiXj: " << t1 << " " << t2 << endl;

                // If Xi and Xj are too far away, then we do not expand the cluster
                // This is to control the noise of approx neighborhoods returned by ANNS solvers
                // If Xj belongs to Xi's cluster, it should be connected via another point Xk, i.e.
                if ( distXiXj > (vec2D_NeighborDist_[Xi][t1 - 1].second + vec2D_NeighborDist_[Xj][t2 - 1].second))
                    bExpandCluster = false;
            }

            if ( bExpandCluster )
            {
                const auto& Xj_neighborhood = vec2D_NeighborDist_[Xj];
                int Xj_neighborSize = static_cast<int>(Xj_neighborhood.size());

                // vector<pair<int, float>> top_KNN(Xj_neighborhood.begin(), Xj_neighborhood.begin() + min(ck, Xj_neighborSize));
                // if (top_KNN.empty())
                //     cout << "Bug in Xj_neighborhood: " << top_KNN.size() << endl;

                // Note: Check one of minPts neighbors has label as the predecessor
                // as we want to spread cluster info via min reachability-dist
                bool hasLabel = false;

                // for (const auto& p : top_KNN)
                for (auto it = Xj_neighborhood.begin(); it != Xj_neighborhood.begin() + min(k_expand, Xj_neighborSize);++it)
                {
                    if (labels[it->first] == predLabel)
                    {
                        hasLabel = true;
                        break;
                    }
                }

                // All kNN points do not have predecessor label, create new cluster
                if ( !hasLabel )
                {
                    clusterId = clusterId + 1;
                    labels[Xj] = clusterId;
                    vecClusters.emplace_back(1);
                }
                else // Use the predecessor's label
                {
                    labels[Xj] = predLabel; // label of predecessor
                    vecClusters[predLabel] += 1;
                }

                // Now we extend the seedSet with Xj_neighborhood
                // Case 1: If Xj starts the new cluster, we tend to process the points around Xj in the new cluster
                // It this is the case, then we might process border points from previous cluster
                // This is why we keep predecessors' label to connect border points to previous cluster.
                // Case 2: If Xj is connected to the old cluster, we also extend PQ with Xj's neighbors
                if (propagation_cutoff)
                    Xj_neighborSize = min(Xj_neighborSize, k_expand);

//                for (auto & p : Xj_neighborhood)
                for (auto it = Xj_neighborhood.begin(); it != Xj_neighborhood.begin() + Xj_neighborSize;++it)
                {
                    const auto& p = *it;

                    int Xk = p.first; // first: point idx, second: dist

                    // only update if it is not processed
                    if (processSet[Xk])
                        continue;

                    // Note: This condition is nice to reduce PQ since we aim at finding min reachability distance
                    if (minConnectedDist[Xk] < p.second)
                        continue;

                    // Heuristic to reduce PQ size: only add to PQ for smaller connected dist(Xi, Xj)
                    minConnectedDist[Xk] = p.second;

                    float weight = 0.0;
                    if ((int)vec2D_NeighborDist_[Xk].size() < k)
                        weight = p.second;
                    else
                        weight = (p.second + vec2D_NeighborDist_[Xk][k - 1].second) / 2;

                    seedSet.emplace(Xk, Xj, weight);

                }
            }
            else
            {
                // Note: If we reset PQ for new cluster Xj, then we mis-classify the border point from previous cluster
                // The cluster quality is significantly decreased
                processSet[Xj] = false;

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
 *
 */
void cluprop::knn_dane(const Ref<const RowMajorMatrixXi> & matIndices, const Ref<const RowMajorMatrixXf> & matDistances, const int k, const int k_expand)
{
    // Form vec2D_NeighborDist from vecIndices and vecDistances
    // sVDC::n_points = vecIndices.size();
    n_points = matIndices.rows();
    int n_neighbors = matIndices.cols();

    // Step 1: Form symmetric kNN graph
    vec2D_NeighborDist_ = vector< vector< pair<int, float> > > (n_points, vector< pair<int, float> >());

    // Note: If NUM_LOCKS is large, we might not have enough stack memory if using array
    // 16K locks is good for million-point data set though it is not good for small data sets.
    constexpr size_t NUM_LOCKS = 16384;
    vector<omp_lock_t> locks(NUM_LOCKS); // NUM_LOCK = 16K locks = only 256 KB
    // Initialize locks
    // #pragma omp parallel for
    for (size_t i = 0; i < NUM_LOCKS; i++) {
        omp_init_lock(&locks[i]);
    }

    // for (int i = 0; i < vecIndices[0].size(); i++) {
    //     cout << vecIndices[0][i] << " " << vecDistances[0][i] << endl;
    // }


    #pragma omp parallel for
    for (int n = 0; n < n_points; n++ ) {
        for (int i = 0; i < n_neighbors; ++i) {
            int iPointIdx = matIndices(n, i); // vecIndices[n][i];
            float fDist = matDistances(n, i); // vecDistances[n][i];

            // Skip if the point is not in the range of [0, n_points)
            if (iPointIdx < 0 || iPointIdx >= n_points || iPointIdx == n)
                continue;

            omp_set_lock(&locks[n % NUM_LOCKS]);
            vec2D_NeighborDist_[n].emplace_back(iPointIdx, fDist); // duplicate at most twice
            omp_unset_lock(&locks[n % NUM_LOCKS]);

            omp_set_lock(&locks[iPointIdx % NUM_LOCKS]);
            vec2D_NeighborDist_[iPointIdx].emplace_back(n, fDist); // so vector is much better than map()
            omp_unset_lock(&locks[iPointIdx % NUM_LOCKS]);
        }
    }

    // Destroy locks
    for (size_t i = 0; i < NUM_LOCKS; i++) {
        omp_destroy_lock(&locks[i]);
    }

    // Sorting vec2D_NeighborDist[n] by distance
    #pragma omp parallel for
    for (int n = 0; n < n_points; ++n) {

        // Step 1: Sort by value (float)
        std::sort(vec2D_NeighborDist_[n].begin(), vec2D_NeighborDist_[n].end(), [](const auto& a, const auto& b) {
            // Compare based on the float value first
            if (a.second != b.second) {
                return a.second < b.second; // Sort by float in ascending order
            }
            // If float values are equal, compare based on the int value
            return a.first < b.first; // Sort by int in ascending order
        });


        // Step 2: Linear scan and merge duplicates
        std::vector<std::pair<int, float>> dedup;
        // dedup.reserve(sVDC::vec2D_NeighborDist[n].size());  // optional optimization

        for (size_t i = 0; i < vec2D_NeighborDist_[n].size(); ++i) {
            if (dedup.empty() || vec2D_NeighborDist_[n][i].first != dedup.back().first) {
                dedup.push_back(vec2D_NeighborDist_[n][i]);
            } else {
                // Keep min distance value (can switch to max or average)
                dedup.back().second = min(dedup.back().second, vec2D_NeighborDist_[n][i].second);
            }
        }

        vec2D_NeighborDist_[n] = dedup;
    }

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
        dane_(k, k);
    else
        dane_(k, k_expand);
}
