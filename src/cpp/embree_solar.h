#pragma once
#include <embree4/rtcore.h>
#include <stdio.h>
#include <math.h>
#include <limits>
#include <iostream>
#include <vector>
#include "common.h"
#include "sunrays.h"
#include "skydome.h"

#ifdef PYTHON_MODULE
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

// using namespace std;

class EmbreeSolar
{

public:
    EmbreeSolar();
    EmbreeSolar(std::vector<std::vector<float>> vertices, std::vector<std::vector<int>> faces);
    virtual ~EmbreeSolar();

    void CreateDevice();
    void CreateScene();
    void CreateGeom(std::vector<std::vector<float>> vertices, std::vector<std::vector<int>> faces);
    void CreateGeomPlane();
    void CalcFaceMidPoints();

    int GetSkydomeRayCount();

    std::vector<std::vector<int>> GetMeshFaces();
    std::vector<std::vector<float>> GetMeshVertices();
    std::vector<std::vector<float>> GetFaceNormals();

    std::vector<std::vector<int>> GetOccludedResults();
    std::vector<std::vector<float>> GetAngleResults();

    std::vector<std::vector<int>> GetFaceSkyHitResults();
    std::vector<float> GetFaceSkyPortionResults();

    std::vector<std::vector<int>> GetSkydomeFaces();
    std::vector<std::vector<float>> GetSkydomeVertices();
    std::vector<std::vector<float>> GetSkydomeRayDirections();

    void Raytrace_occ1(std::vector<float> &angles, std::vector<int> &occluded, int &hitCounter);
    void Raytrace_occ4(std::vector<float> &angles, std::vector<int> &occluded, int &hitCounter);
    void Raytrace_occ8(std::vector<float> &angles, std::vector<int> &occluded, int &hitCounter);
    void Raytrace_occ16(std::vector<float> &angles, std::vector<int> &occluded, int &hitCounter);

    bool SunRaytrace_Occ1(std::vector<std::vector<float>> sun_vecs);
    bool SunRaytrace_Occ4(std::vector<std::vector<float>> sun_vecs);
    bool SunRaytrace_Occ8(std::vector<std::vector<float>> sun_vecs);
    bool SunRaytrace_Occ16(std::vector<std::vector<float>> sun_vecs);

    bool SkyRaytrace_Occ1();
    bool SkyRaytrace_Occ4();
    bool SkyRaytrace_Occ8();
    bool SkyRaytrace_Occ16();

    void CalcFaceNormals();
    void ErrorFunction(void *userPtr, enum RTCError error, const char *str);

private:
    Skydome *mSkydome = NULL;
    Sunrays *mSunrays = NULL;

    RTCScene mScene;
    RTCDevice mDevice;
    RTCGeometry mGeometry;

    Parameters mPp; // plane parameters
    // Parameters mRp; // ray parameters

    int mVertexCount;
    int mFaceCount;

    Face *mFaces;
    Vertex *mVertices;
    Vertex *mFaceMidPts;
    Vector *mFaceNormals;

    // Results from analysis
    std::vector<std::vector<int>> mOccluded;
    std::vector<std::vector<float>> mAngles;

    std::vector<std::vector<int>> mFaceSkyHit;
    std::vector<float> mFaceSkyPortion;
};
