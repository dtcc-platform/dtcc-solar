#pragma once
#include <cmath>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#define PYTHON_MODULE

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using fArray2D = std::vector<std::vector<float>>;
using iArray2D = std::vector<std::vector<int>>;
using fArray1D = std::vector<float>;
using iArray1D = std::vector<int>;
using bArray1D = std::vector<bool>;
using Eigen::MatrixXf;
using Eigen::VectorXf;
using MatrixXfRM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Use typedef instead of using-alias for MSVC compatibility
typedef std::chrono::high_resolution_clock hrClock;
typedef std::chrono::duration<float> fDuration;

#define EIGEN_USE_THREADS

/* vertex and triangle layout */
struct Vertex
{
    float x, y, z;
};

struct Vector
{
    float x, y, z;
};

struct Face
{
    int v0, v1, v2;
};

struct Parameters
{
    float xMin, xMax;
    float yMin, yMax;
    float xPadding, yPadding;
    int xCount, yCount;
};

enum class ResType
{
    hit,
    angle,
};

static inline Vertex CreateVertex(std::vector<float> v)
{
    Vertex vertex = {v[0], v[1], v[2]};
    return vertex;
};

static inline Vector CreateVector(Vertex from, Vertex to)
{
    float x, y, z;
    x = to.x - from.x;
    y = to.y - from.y;
    z = to.z - from.z;
    Vector v = {x, y, z};
    return v;
};

static inline float VectorLength(Vector v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
};

static inline Vector UnitizeVector(Vector v)
{
    float len = std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    Vector vOut = {v.x / len, v.y / len, v.z / len};
    return vOut;
};

static inline std::vector<float> UnitizeVector(std::vector<float> v)
{
    float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    std::vector<float> vOut = {v[0] / len, v[1] / len, v[2] / len};
    return vOut;
};

static inline Vector CrossProduct(Vector a, Vector b)
{
    Vector c;
    c.x = a.y * b.z - a.z * b.y;
    c.y = a.z * b.x - a.x * b.z;
    c.z = a.x * b.y - a.y * b.x;
    return c;
};

static inline Vector DotProduct(Vector a, Vector b)
{
    Vector c;
    c.x = a.x * b.x;
    c.y = a.y * b.y;
    c.z = a.z * b.z;
    return c;
};

static inline float VectorAngle(std::vector<float> v1, std::vector<float> v2)
{
    // Assume that v1 and v2 are unitized.
    float dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    float angle = std::acos(dot);
    return angle;
};

static inline float CalcAngle(Vector ray, Vector nml)
{
    // With unit vectors
    float dot = ray.x * nml.x + ray.y * nml.y + ray.z * nml.z;
    float angleInRadians = std::acos(dot);
    return angleInRadians;
}

static inline float CalcAngle2(Vector ray, Vector nml)
{
    // If vectors are not unitized
    float dot = ray.x * nml.x + ray.y * nml.y + ray.z * nml.z;

    float magN = std::sqrt(nml.x * nml.x + nml.y * nml.y + nml.z * nml.z);
    float magR = std::sqrt(ray.x * ray.x + ray.y * ray.y + ray.z * ray.z);

    // Check for zero vectors
    if (magR == 0.0f || magN == 0.0f)
        throw std::invalid_argument("Zero vector is not allowed.");

    float cosTheta = dot / (magN * magR);

    // Clamp cosTheta to the [-1, 1] range
    cosTheta = std::max(-1.0f, std::min(1.0f, cosTheta));

    float angleInRadians = std::acos(cosTheta);
    return angleInRadians;
}

static inline float Deg2Rad(float deg)
{
    return deg * M_PI / 180.0;
}

static inline float Rad2Deg(float rad)
{
    return rad * 180.0 / M_PI;
}

static inline std::vector<float> Spherical2Cartesian(float r, float elevation, float azimuth)
{
    float x = r * cos(elevation) * cos(azimuth);
    float y = r * cos(elevation) * sin(azimuth);
    float z = r * sin(elevation);
    return {x, y, z};
}

template <typename T>
static inline std::pair<int, int> GetShape(const std::vector<std::vector<T>> &array2D)
{
    int rows = array2D.size();
    int cols = rows > 0 ? array2D[0].size() : 0;
    return {rows, cols};
}

template <typename T>
static inline std::pair<int, int> GetShapeMaxCols(const std::vector<std::vector<T>> &array2D)
{
    int rows = array2D.size();
    int maxCols = 0;
    for (const auto &row : array2D)
    {
        if (row.size() > maxCols)
        {
            maxCols = row.size();
        }
    }
    return {rows, maxCols};
}

static inline MatrixXf VectorToEigen(const std::vector<std::vector<float>> &vec)
{
    if (vec.empty() || vec[0].empty())
        throw std::runtime_error("Empty input matrix");

    size_t rows = vec.size();
    size_t cols = vec[0].size();

    MatrixXf mat(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            mat(i, j) = vec[i][j];

    return mat;
}

static inline MatrixXfRM VectorToEigenRM(const std::vector<std::vector<float>> &vec)
{
    if (vec.empty() || vec[0].empty())
        throw std::runtime_error("Empty input matrix");

    size_t rows = vec.size();
    size_t cols = vec[0].size();

    MatrixXfRM mat(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            mat(i, j) = vec[i][j];

    return mat;
}

static inline fArray2D EigenToVector(const Eigen::MatrixXf &mat)
{
    fArray2D result(mat.rows(), std::vector<float>(mat.cols()));

    for (int i = 0; i < mat.rows(); ++i)
    {
        for (int j = 0; j < mat.cols(); ++j)
        {
            result[i][j] = mat(i, j);
        }
    }
    return result;
}

static inline fArray2D EigenToVectorRM(const MatrixXfRM &mat)
{
    fArray2D result(mat.rows(), std::vector<float>(mat.cols()));

    for (int i = 0; i < mat.rows(); ++i)
    {
        for (int j = 0; j < mat.cols(); ++j)
        {
            result[i][j] = mat(i, j);
        }
    }
    return result;
}