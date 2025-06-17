#pragma once
#include <embree4/rtcore.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <omp.h>
#include <thread>
#include <stdio.h>
#include <math.h>
#include <limits>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include "common.h"
#include "sunrays.h"
#include "skydome.h"
#include "pydome.h"
#include "logging.h"

#ifdef PYTHON_MODULE
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

class EmbreeSolar
{

public:
    EmbreeSolar();
    EmbreeSolar(fArray2D vertices, iArray2D faces);
    EmbreeSolar(fArray2D vertices, iArray2D faces, std::vector<bool> face_mask);
    EmbreeSolar(fArray2D vertices, iArray2D faces, std::vector<bool> face_mask, int skyType);
    EmbreeSolar(fArray2D vertices, iArray2D faces, std::vector<bool> face_mask, fArray2D rays, fArray1D areas);
    virtual ~EmbreeSolar();

    void CreateDevice();
    void CreateScene();
    void CreateGeom(fArray2D vertices, iArray2D faces);
    void CreateGeomPlane();
    void CalcFaceMidPoints();

    int GetSkydomeRayCount();

    iArray2D GetMeshFaces();
    fArray2D GetMeshVertices();
    fArray2D GetFaceNormals();

    iArray2D GetOccludedResults();
    fArray2D GetAngleResults();

    iArray2D GetFaceSkyHitResults();
    fArray1D GetSkyViewFactorResults();

    iArray2D GetSkydomeFaces();
    fArray2D GetSkydomeVertices();
    fArray2D GetSkydomeRayDirections();

    fArray2D GetPydomeRayDirections();
    fArray2D GetVisibilityResults();
    fArray2D GetProjectionResults();
    fArray2D GetIrradianceResults();

    std::vector<float> GetAccumulatedAngles();
    std::vector<float> GetAccumulatedOcclusion();

    void Raytrace_occ1(std::vector<float> &angles, std::vector<int> &occluded, int &hitCounter);
    void Raytrace_occ8(std::vector<float> &angles, std::vector<int> &occluded, int &hitCounter);

    bool SunRaytrace_Occ1(fArray2D sun_vecs);
    bool SunRaytrace_Occ8(fArray2D sun_vecs);

    bool SkyRaytrace_Occ1();
    bool SkyRaytrace_Occ8();

    bool CalcProjMatrix();
    bool CalcVisMatrix_Occ1();
    bool CalcVisMatrix_Occ8();
    bool CalcVisProjMatrix();
    bool CalcIrradiance(fArray2D arr);
    bool Run2PhaseAnalysis(fArray2D sunSkyMat);

    bool Accumulate();

    void CalcFaceNormals();
    void ErrorFunction(void *userPtr, enum RTCError error, const char *str);

private:
    Skydome *mSkydome = NULL;
    Sunrays *mSunrays = NULL;
    Pydome *mPydome = NULL;

    RTCScene mScene;
    RTCDevice mDevice;
    RTCGeometry mGeometry;

    Parameters mPp; // plane parameters

    int mVertexCount;
    int mFaceCount;

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
    bool mHasVisResults;
    bool mHasProjResults;
    bool mHasVisProjResults;

    iArray2D mOccluded;
    fArray2D mAngles;

    std::vector<float> mAccumAngles;
    std::vector<float> mAccumOcclud;

    iArray2D mFaceSkyHit;
    std::vector<float> mSkyViewFactor;

    fArray2D mProjectionMatrix;
    fArray2D mVisibilityMatrix;
    fArray2D mVisProjMatrix;
    fArray2D mIrradianceMatrix;
};
