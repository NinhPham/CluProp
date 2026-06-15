
#ifndef CLUPROP_H
#define CLUPROP_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include "header.h"

class cluprop {

public:

    // RowMajorMatrixXf matrix_X; // public as we will have to load data into it (when data is big)

    int n_points;
    int n_features;

    // used on DANE to cut off neighbors that are too far away when inserting into the priority queue, default is false.
    // This is to reduce the number of neighbors to be extended, and improve efficiency.
    // If this flag is true, then we only extend neighbors to min(neighborSize, c * minPts),
    // where neighborSize is the current number of neighbors found, and c is a constant (default 1)
    bool propagation_cutoff = false;

    // minimum cluster size for cluster expansion - we always expand clusters if the cluster does not have min_cluster_size points
    // When it has enough points, we will only expand if dist(Xi, Xj) < kNN_dist(Xi) + kNN_dist(Xj) - to ensure cluster spreads slowly from dense region to sparse region
    // This is to avoid large clusters that cover most points
    int min_cluster_size = 50;

    int n_threads = 8;
    bool verbose = false;
    string output;

private:

    // Data structures of DNP
    vector< vector< pair<int, float> > > vec2D_NeighborDist_; // vector of neighborhoods and its distances from the graph

public:

    // Clustering's output
    IVector labels;
    int n_clusters = 0;

    cluprop(int n, int d){
        n_points = n;
        n_features = d;
    }

    void set_prop_params(bool ver = false, string filename = "", int minClusterSize = 50){
        verbose = ver;

        // Current not support multi-thread
        // set_threads(numThreads);

        output = filename;
        min_cluster_size = minClusterSize;
    }



    void clear(){

        n_clusters = 0;
        labels.clear();
        vec2D_NeighborDist_.clear(); // vector of approx neighborhoods and its distances
    }

    ~cluprop(){
        clear();
    }


    void set_min_cluster_size(float s){ min_cluster_size = s; }
    void set_propagation_cutoff(bool b){ propagation_cutoff = b; }

    void set_threads(int t)
    {
        if (t <= 0)
            #ifdef _OPENMP
                n_threads = omp_get_max_threads();
            #else
                n_threads = 1;
            #endif
        else
            n_threads = t;
    }

    // DANE with precomputed kNN
    void knn_dane(const Ref<const RowMajorMatrixXi> & , const Ref<const RowMajorMatrixXf> & , int, int = -1);


    // Placeholder
    // void cluprop::fit(const Ref<const RowMajorMatrixXf> & MATRIX_X, const string& knn_alg, int k, float c);

private:

    void dane_(int, int);
};


#endif // CLUPROP_H
