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
#include "rays.h"
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
    EmbreeSolar(fArray2D vertices, iArray2D faces, std::vector<bool> face_mask, fArray2D sunSkyRays, fArray1D solidAngles);
    EmbreeSolar(fArray2D vertices, iArray2D faces, std::vector<bool> face_mask, fArray2D skyRays, fArray1D solidAngles, fArray2D sunRays);
    virtual ~EmbreeSolar();

    void CreateDevice();
    void CreateScene();
    void CreateGeom(fArray2D vertices, iArray2D faces);
    void CreateGeomPlane();
    void CalcFaceMidPoints();
    void CalcFaceNormals();

    iArray2D GetMeshFaces();
    fArray2D GetMeshVertices();
    fArray2D GetFaceNormals();

    fArray2D GetVisibilityMatrixTot();
    fArray2D GetProjectionMatrixTot();
    fArray2D GetIrradianceMatrixTot();

    fArray2D GetVisibilityMatrixSky();
    fArray2D GetProjectionMatrixSky();
    fArray2D GetIrradianceMatrixSky();

    fArray2D GetVisibilityMatrixSun();
    fArray2D GetProjectionMatrixSun();
    fArray2D GetIrradianceMatrixSun();

    bool CalcProjMatrix(Rays *rays, fArray2D &projMatrix);
    bool CalcVisMatrix_Occ1(Rays *rays, fArray2D &visMatrix);
    bool CalcVisMatrix_Occ8(Rays *rays, fArray2D &visMatrix);
    bool CalcVisProjMatrix(Rays *rays, fArray2D &visMatrix, fArray2D &projMatrix, fArray2D &visProjMatrix);

    bool CalcIrradiance2Phase(Rays *rays, fArray2D &skySunMatrix, fArray2D &visProjMatrix, fArray2D &irrMatrix);
    bool CalcIrradiance3Phase(Rays *skyRays, Rays *sunRays, fArray2D &skyMatrix, fArray2D &sunMatrix, fArray2D &skyVisProjMatrix, fArray2D &sunVisProjMatrix, fArray2D &skyIrrMatrix, fArray2D &sunIrrMatrix);

    bool Run2PhaseAnalysis(fArray2D sunSkyMatrix);
    bool Run3PhaseAnalysis(fArray2D skyMatrix, fArray2D sunMatrix);

    void ErrorFunction(void *userPtr, enum RTCError error, const char *str);

private:
    // Skydome *mSkydome = NULL;
    // Sunrays *mSunrays = NULL;

    Rays *mSunSkyRays = NULL;
    Rays *mSkyRays = NULL;
    Rays *mSunRays = NULL;

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

    iArray2D mOccluded;
    fArray2D mAngles;

    std::vector<float> mAccumAngles;
    std::vector<float> mAccumOcclud;

    iArray2D mFaceSkyHit;
    std::vector<float> mSkyViewFactor;

    fArray2D mProjMatrixTot;
    fArray2D mVisMatrixTot;
    fArray2D mVisProjMatrixTot;
    fArray2D mIrrMatrixTot;

    fArray2D mProjMatrixSky;
    fArray2D mVisMatrixSky;
    fArray2D mVisProjMatrixSky;
    fArray2D mIrrMatrixSky;

    fArray2D mProjMatrixSun;
    fArray2D mVisMatrixSun;
    fArray2D mVisProjMatrixSun;
    fArray2D mIrrMatrixSun;

    float mDomeSolidAngle = 2 * M_PI; // Solid angle of the dome, 2 * pi steradians
};
