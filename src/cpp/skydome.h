#pragma once
#include </opt/homebrew/Cellar/embree/4.3.0/include/embree4/rtcore.h>
#include <stdio.h>
#include <math.h>
#include <limits>
#include <iostream>
#include <vector>
#include "common.h"

class Skydome
{

public:
    Skydome();
    Skydome(int rayCount);
    ~Skydome();

    void InitRays(int rayCount);
    void CreateMesh(int nStrips);
    void CreateRays();
    void BundleRays();

    void TranslateRays(Vertex new_origin);
    void Translate4Rays(Vertex new_origin);
    void Translate8Rays(Vertex new_origin);
    void Translate16Rays(Vertex new_origin);

    void Raytrace();

    float CalcSphereCapArea(float elevation);
    float CalcHemisphereArea();

    void GetTopCap(float maxElev, float elev, float maxAzim, int nAzim, float faceAreaPart);
    void CreateTopCapQuads(float elev, float nextElev, float azim, float d_azim);
    void CreateDomeMeshQuad(float azim, float nextAzim, float elev, float nextElev);

    float CalcSphereStripArea(float elev1, float elev2);

    int GetRayCount();
    int GetBundle4Count();
    int GetBundle8Count();
    int GetBundle16Count();

    RTCRay *GetRays();
    RTCRay4 *GetRays4();
    RTCRay8 *GetRays8();
    RTCRay16 *GetRays16();

    int **GetValid4();
    int **GetValid8();
    int **GetValid16();

    std::vector<std::vector<int>> GetFaces();
    std::vector<std::vector<float>> GetVertices();
    std::vector<std::vector<float>> GetRayDirections();
    std::vector<float> GetRayAreas();

private:
    int mRayCount;
    int mBundle4Count;
    int mBundle8Count;
    int mBundle16Count;

    // Skydome mesh entities
    std::vector<std::vector<int>> mFaces;
    std::vector<std::vector<float>> mVertices;

    // Skydome ray data
    std::vector<float> mRayOrigin;
    std::vector<float> mRayAreas;
    std::vector<std::vector<float>> mRayDirections;

    RTCRay *mRays;
    RTCRay4 *mRays4;
    RTCRay8 *mRays8;
    RTCRay16 *mRays16;

    int **mRays4Valid;
    int **mRays8Valid;
    int **mRays16Valid;
};
