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
    void UpdateRay4Directions(std::vector<float> new_sun_vec, bool applyMask);
    void UpdateRay8Directions(std::vector<float> new_sun_vec, bool applyMask);
    void UpdateRay16Directions(std::vector<float> new_sun_vec, bool applyMask);

    int GetRayCount();
    int GetBundle4Count();
    int GetBundle8Count();
    int GetBundle16Count();

    std::vector<RTCRay> &GetRays();
    std::vector<RTCRay4> &GetRays4();
    std::vector<RTCRay8> &GetRays8();
    std::vector<RTCRay16> &GetRays16();

    std::vector<RTCRayHit> &GetRayHit();
    std::vector<RTCRayHit4> &GetRayHit4();
    std::vector<RTCRayHit8> &GetRayHit8();
    std::vector<RTCRayHit16> &GetRayHit16();

    int **GetValid4(bool applyMask);
    int **GetValid8(bool applyMask);
    int **GetValid16(bool applyMask);

private:
    int mRayCount;
    int mBundle4Count;
    int mBundle8Count;
    int mBundle16Count;

    Parameters mRp; // ray parameters

    // Skydome ray data
    std::vector<float> mRayOrigin;
    std::vector<float> mRayAreas;
    std::vector<std::vector<float>> mRayDirections;

    std::vector<RTCRay> mRays;
    std::vector<RTCRay4> mRays4;
    std::vector<RTCRay8> mRays8;
    std::vector<RTCRay16> mRays16;

    int **mRays4Valid;
    int **mRays8Valid;
    int **mRays16Valid;

    int **mRays4ValidMask;
    int **mRays8ValidMask;
    int **mRays16ValidMask;

    std::vector<bool> mFaceMask;
};
