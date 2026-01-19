#pragma once
#include "bvh_types.h"
#include <stdio.h>
#include <math.h>
#include <limits>
#include <iostream>
#include <vector>
#include <algorithm>
#include "common.h"
#include "logging.h"

class Sunrays
{
public:
    Sunrays();
    Sunrays(Vertex *faceMidPoints, int faceCount, std::vector<bool> faceMask);
    ~Sunrays();

    void InitRays(int rayCount);
    void CreateGridRays();

    void CreateRays(Vertex *faceMidPts, int faceCount);

    void UpdateRayDirections(std::vector<float> new_sun_vec, bool applyMask);

    int GetRayCount();

    std::vector<Ray> &GetRays();

private:
    int mRayCount;

    Parameters mRp; // ray parameters

    // Sunray data
    std::vector<float> mRayOrigin;
    std::vector<float> mRayAreas;
    std::vector<std::vector<float>> mRayDirections;

    std::vector<Ray> mRays;

    std::vector<bool> mFaceMask;
};
