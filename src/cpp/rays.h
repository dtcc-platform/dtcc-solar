#pragma once
#include "common.h"
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
    Rays(fArray2D rays, fArray1D solidAngles);
    ~Rays();

    void InitRays(fArray2D rays);
    void CreateRays();

    const int GetRayCount() const;
    const std::vector<Ray> &GetRays() const;
    const fArray2D &GetRayDirections() const;
    const fArray1D &GetSolidAngles() const;
    float GetDomeSolidAngle() const;

private:
    int mRayCount;

    // Skydome ray data
    std::vector<float> mRayOrigin;
    std::vector<std::vector<float>> mRayDirections;

    // Ray areas / tot area
    std::vector<float> mRaySolidAngles;

    std::vector<Ray> mRays;
};
