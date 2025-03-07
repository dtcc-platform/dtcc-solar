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

class Skydome
{

public:
    Skydome();
    Skydome(int skyType);
    ~Skydome();

    void InitRays(int rayCount);
    void CreateEqualAreaMesh(int nStrips);
    void CreateTregenzaMesh();
    void CreateReinhartMesh();
    void CreateRays();
    void BundleRays();

    void TranslateRays(Vertex new_origin);
    void Translate4Rays(Vertex new_origin);
    void Translate8Rays(Vertex new_origin);
    void Translate16Rays(Vertex new_origin);

    float CalcSphereCapArea(float elevation);
    float CalcHemisphereArea();

    void GetTopCap(float maxElev, float elev, float maxAzim, int nAzim, float faceAreaPart);
    void CreateMeshQuad(float azim, float nextAzim, float elev, float nextElev);

    void CreateEqualAreaZenithPatch(float elev, float nextElev, float azim, float d_azim);
    void CreateTregenzaZenithPatch(float elevation);
    void CreateReinhartZenithPatch(float elevation);

    float CalcSphereStripArea(float elev1, float elev2);
    float CalcSpherePatchArea(float r, float elev1, float elev2, float azim1, float azim2);

    int GetFaceCount();
    int GetRayCount();
    int GetBundle4Count();
    int GetBundle8Count();
    int GetBundle16Count();

    std::vector<RTCRay> &GetRays();
    std::vector<RTCRay4> &GetRays4();
    std::vector<RTCRay8> &GetRays8();
    std::vector<RTCRay16> &GetRays16();

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

    std::vector<RTCRay> mRays;
    std::vector<RTCRay4> mRays4;
    std::vector<RTCRay8> mRays8;
    std::vector<RTCRay16> mRays16;

    int **mRays4Valid;
    int **mRays8Valid;
    int **mRays16Valid;
};
