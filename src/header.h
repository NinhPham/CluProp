#pragma once

#include <Eigen/Dense>
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <queue>
#include <random>
#include <algorithm>
#include <list>
#include <utility>
#include <limits>

#include <chrono>
#include <iostream> // cin, cout

//#include <boost/multi_array.hpp>
#include <boost/dynamic_bitset.hpp>

#define PI				3.141592653589793238460
#define POS_INF std::numeric_limits<float>::infinity()
#define NEG_INF (-std::numeric_limits<float>::infinity())
#define EPSILON         0.000001


using namespace Eigen;
using namespace std;

typedef vector<float> FVector;
typedef vector<int> IVector;

using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMajorMatrixXi = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

//typedef vector<uint32_t> I32Vector;
//typedef vector<uint64_t> I64Vector;


//typedef boost::multi_array<int, 3> IVector3D;

//struct myComp
//{
//
//    constexpr bool operator()(
//        pair<double, int> const& a,
//        pair<double, int> const& b)
//    const noexcept
//    {
//        return a.first > b.first;
//    }
//};
struct IFPair
{
    int m_iIndex;
    float	m_fValue;

    IFPair()
    {
        m_iIndex = 0;
        m_fValue = 0.0;
    }

    IFPair(int p_iIndex, float p_fValue)
    {
        m_iIndex = p_iIndex;
        m_fValue = p_fValue;
    }

    // Overwrite operation < to get top K largest entries
    bool operator<(const IFPair& p) const
    {
        return m_fValue < p.m_fValue;
    }

    bool operator>(const IFPair& p) const
    {
        return m_fValue > p.m_fValue;
    }
};

struct IIFTriple
{
    int m_iIndex;    // Point idx
    int m_iPred;  // Predecessor idx
    float	m_fValue;

    IIFTriple()
    {
        m_iIndex = 0;
        m_iPred = 0;
        m_fValue = 0.0;
    }

    IIFTriple(int p_iIndex, int p_iPred, float p_fValue)
    {
        m_iIndex = p_iIndex;
        m_iPred = p_iPred;
        m_fValue = p_fValue;
    }

    // Overwrite operation < to get top K largest entries
    bool operator<(const IIFTriple& p) const
    {
        return m_fValue < p.m_fValue;
    }

    bool operator>(const IIFTriple& p) const
    {
        return m_fValue > p.m_fValue;
    }
};

// typedef priority_queue<IFPair, vector<IFPair>, greater<IFPair>> Min_PQ_Pair;
// typedef priority_queue<IFPair, vector<IFPair>, less<IFPair>> Max_PQ_Pair;
typedef priority_queue<IIFTriple, vector<IIFTriple>, greater<IIFTriple>> Min_PQ_Triple;

