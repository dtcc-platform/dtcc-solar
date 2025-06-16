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

class Pydome
{

public:
    Pydome(fArray2D rays, fArray1D areas);
    ~Pydome();

    void InitRays();
    void CreateRays();
    void BundleRays();

    void TranslateRays(Vertex new_origin);
    void Translate8Rays(Vertex new_origin);

    int GetRayCount();
    int GetBundle8Count();

    std::vector<RTCRay> &GetRays();
    std::vector<RTCRay8> &GetRays8();

    int **GetValid8();

    fArray2D GetRayDirections();
    fArray1D GetSolidAngles();
    float GetDomeSolidAngle();

private:
    int mRayCount;
    int mBundle8Count;

    // Skydome ray data
    std::vector<float> mRayOrigin;
    std::vector<std::vector<float>> mRayDirections;

    // Ray areas / tot area
    std::vector<float> mRaySolidAngle;

    std::vector<RTCRay> mRays;
    std::vector<RTCRay8> mRays8;

    float mDomeSolidAngle = 2 * M_PI; // Solid angle of the dome, 2 * pi steradians

    int **mRays8Valid;
};