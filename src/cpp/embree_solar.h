#pragma once
#include <embree4/rtcore.h>
#include <stdio.h>
#include <math.h>
#include <limits>
#include <iostream>
#include <chrono>
#include <vector>
#include "common.h"
#include "sunrays.h"
#include "skydome.h"
#include "logging.h"

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
    EmbreeSolar(std::vector<std::vector<float>> vertices, std::vector<std::vector<int>> faces, std::vector<bool> face_mask);
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

    std::vector<float> GetIrradianceResultsDNI();
    std::vector<float> GetIrradianceResultsDHI();
    std::vector<float> GetAccumulatedAngles();
    std::vector<float> GetAccumulatedOcclusion();

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

    bool CalcIrradiance(std::vector<float> dni, std::vector<float> dhi);
    bool CalcIrradianceGroup(std::vector<float> dni, std::vector<float> dhi, std::vector<std::vector<int>> sunGroups);

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

    size_t mVertexCount;
    size_t mFaceCount;

    Face *mFaces;
    Vertex *mVertices;
    Vertex *mFaceMidPts;
    Vector *mFaceNormals;

    int mMaskCount;
    bool mApplyMask;
    std::vector<bool> mFaceMask;

    bool mHasSunResults;
    bool mHasSkyResults;
    bool mHasIrrResults;

    // Results from analysis based on sun vectors or group sun vectors
    std::vector<std::vector<int>> mOccluded;
    std::vector<std::vector<float>> mAngles;

    // Results from analysis mapped onto all sun positions
    std::vector<float> mIrradianceDNI;
    std::vector<float> mIrradianceDHI;
    std::vector<float> mAccumAngles;
    std::vector<float> mAccumOcclud;

    std::vector<std::vector<int>> mFaceSkyHit;
    std::vector<float> mFaceSkyPortion;
};
