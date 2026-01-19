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

    void TranslateRays(Vertex new_origin);

    float CalcSphereCapArea(float elevation);
    float CalcHemisphereArea();

    void CreateMeshQuad(float azim, float nextAzim, float elev, float nextElev);

    void CreateTregenzaZenithPatch(float elevation);
    void CreateReinhartZenithPatch(float elevation);

    float CalcSphereStripArea(float elev1, float elev2);
    float CalcSpherePatchArea(float r, float elev1, float elev2, float azim1, float azim2);

    int GetFaceCount();
    int GetRayCount();

    std::vector<Ray> &GetRays();

    std::vector<std::vector<int>> GetFaces();
    std::vector<std::vector<float>> GetVertices();
    std::vector<std::vector<float>> GetRayDirections();
    std::vector<float> GetRayAreas();

private:
    int mRayCount;

    // Skydome mesh entities
    std::vector<std::vector<int>> mFaces;
    std::vector<std::vector<float>> mVertices;

    // Skydome ray data
    std::vector<float> mRayOrigin;
    std::vector<float> mRayAreas;
    std::vector<std::vector<float>> mRayDirections;

    std::vector<Ray> mRays;
};
