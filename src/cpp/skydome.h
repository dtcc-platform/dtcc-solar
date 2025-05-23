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

    void CreateMeshQuad(float azim, float nextAzim, float elev, float nextElev);

    void CreateTregenzaZenithPatch(float elevation);
    void CreateReinhartZenithPatch(float elevation);

    float CalcSphereStripArea(float elev1, float elev2);
    float CalcSpherePatchArea(float r, float elev1, float elev2, float azim1, float azim2);

    int GetFaceCount();
    int GetRayCount();
    int GetBundle8Count();

    std::vector<RTCRay> &GetRays();
    std::vector<RTCRay8> &GetRays8();

    int **GetValid8();

    std::vector<std::vector<int>> GetFaces();
    std::vector<std::vector<float>> GetVertices();
    std::vector<std::vector<float>> GetRayDirections();
    std::vector<float> GetRayAreas();

private:
    int mRayCount;
    int mBundle8Count;

    // Skydome mesh entities
    std::vector<std::vector<int>> mFaces;
    std::vector<std::vector<float>> mVertices;

    // Skydome ray data
    std::vector<float> mRayOrigin;
    std::vector<float> mRayAreas;
    std::vector<std::vector<float>> mRayDirections;

    std::vector<RTCRay> mRays;
    std::vector<RTCRay8> mRays8;

    int **mRays8Valid;
};
