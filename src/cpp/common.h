#pragma once
#include <cmath>
#define PYTHON_MODULE

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

static inline Vector CreateVector(Vertex from, Vertex to)
{
    float x, y, z;
    x = to.x - from.x;
    y = to.y - from.y;
    z = to.z - from.z;
    Vector v = {x, y, z};
    return v;
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

static inline float VectorAngle(std::vector<float> v1, std::vector<float> v2)
{
    // Assume that v1 and v2 are unitized.
    float dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    float angle = std::acos(dot);
    return angle;
};
