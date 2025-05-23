#pragma once
#include <embree4/rtcore.h>
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
    void BundleRays();

    void UpdateRay1Directions(std::vector<float> new_sun_vec, bool applyMask);
    void UpdateRay8Directions(std::vector<float> new_sun_vec, bool applyMask);

    int GetRayCount();
    int GetBundle8Count();

    std::vector<RTCRay> &GetRays();
    std::vector<RTCRay8> &GetRays8();

    std::vector<RTCRayHit> &GetRayHit();
    std::vector<RTCRayHit8> &GetRayHit8();

    int **GetValid8(bool applyMask);

private:
    int mRayCount;
    int mBundle8Count;

    Parameters mRp; // ray parameters

    // Skydome ray data
    std::vector<float> mRayOrigin;
    std::vector<float> mRayAreas;
    std::vector<std::vector<float>> mRayDirections;

    std::vector<RTCRay> mRays1;
    std::vector<RTCRay8> mRays8;

    int **mRays8Valid;
    int **mRays8ValidMask;

    std::vector<bool> mFaceMask;
};
