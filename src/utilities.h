#ifndef UTILITIES_H_INCLUDED
#define UTILITIES_H_INCLUDED

#include "header.h"

#include <sstream> // stringstream
#include <time.h> // for time(0) to generate different random number
#include <omp.h>
#include <queue>

#define ASSERT_RELEASE(cond, msg)                        \
do {                                                 \
if (!(cond)) {                                   \
std::cerr << "Assertion failed: " << msg     \
<< " (" << __FILE__ << ":"         \
<< __LINE__ << ")\n";              \
std::abort();                                \
}                                                \
} while (0)

/**
Convert an integer to string
**/
inline string int2str(int x)
{
    stringstream ss;
    ss << x;
    return ss.str();
}

/**
Get sign
**/
inline int sgn(float x)
{
    if (x >= 0) return 1;
    else return -1;
    // return 0;
}

inline float compute_reachability(const float kNN_Xi, const float kNN_Xj, const float distXiXj, const string reachDistType) {
    if (reachDistType == "DANE") {
        return (kNN_Xj + distXiXj) / 2;
    }
    else if (reachDistType == "OPTICS") {
        return max(kNN_Xi, distXiXj);
    }
    else if (reachDistType == "HDBSCAN") {
        return max({kNN_Xi, kNN_Xj, distXiXj});
    }
    else
        return distXiXj;
}


// Saving
void outputLabels(const IVector &, const string&);

// Parsing input and param
void loadtxtData(const string&, const string&, int , int, RowMajorMatrixXf & ); // load data fron filename
void loadbinData(const string& , const string& , int , int , RowMajorMatrixXf &);
void transformData(RowMajorMatrixXf & , const string& );

#endif // UTILITIES_H_INCLUDED
