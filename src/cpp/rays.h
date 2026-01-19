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

class Rays
{

public:
    Rays(fArray2D rays);
    Rays(fArray2D rays, fArray1D areas);
    ~Rays();

    void InitRays(fArray2D rays);
    void CreateRays();

    void TranslateRays(Vertex new_origin);

    int GetRayCount();

    std::vector<Ray> &GetRays();

    fArray2D GetRayDirections();
    fArray1D GetSolidAngles();
    float GetDomeSolidAngle();

private:
    int mRayCount;

    // Skydome ray data
    std::vector<float> mRayOrigin;
    std::vector<std::vector<float>> mRayDirections;

    // Ray areas / tot area
    std::vector<float> mRaySolidAngles;

    std::vector<Ray> mRays;
};
