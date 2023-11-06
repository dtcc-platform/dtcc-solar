#pragma once
#include </opt/homebrew/Cellar/embree/4.3.0/include/embree4/rtcore.h>
#include <stdio.h>
#include <math.h>
#include <limits>
#include <iostream>
#include <vector>
#include "common.h"

class Sunrays
{

public:
    Sunrays();
    Sunrays(Vertex *faceMidPoints, int faceCount);
    ~Sunrays();

    void InitRays(int rayCount);
    void CreateGridRays();

    void CreateRays();
    void CreateRays(Vertex *faceMidPts, int faceCount);
    void CreateRayHit(Vertex *faceMidPts, int faceCount);

    void BundleRays();
    void BundleRayHit();

    void UpdateRay1Directions(std::vector<float> new_sun_vec);
    void UpdateRay4Directions(std::vector<float> new_sun_vec);
    void UpdateRay8Directions(std::vector<float> new_sun_vec);
    void UpdateRay16Directions(std::vector<float> new_sun_vec);

    void UpdateRayHit8Directions(std::vector<float> new_sun_vec);

    int GetRayCount();
    int GetBundle4Count();
    int GetBundle8Count();
    int GetBundle16Count();

    RTCRay *GetRays();
    RTCRay4 *GetRays4();
    RTCRay8 *GetRays8();
    RTCRay16 *GetRays16();

    RTCRayHit *GetRayHit();
    RTCRayHit4 *GetRayHit4();
    RTCRayHit8 *GetRayHit8();
    RTCRayHit16 *GetRayHit16();

    int **GetValid4();
    int **GetValid8();
    int **GetValid16();

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

    RTCRay *mRays;
    RTCRay4 *mRays4;
    RTCRay8 *mRays8;
    RTCRay16 *mRays16;

    RTCRayHit *mRayHit;
    RTCRayHit4 *mRayHit4;
    RTCRayHit8 *mRayHit8;
    RTCRayHit16 *mRayHit16;

    int **mRays4Valid;
    int **mRays8Valid;
    int **mRays16Valid;
};