#pragma once
#include "bvh_types.h"
#include "Accel.h"

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
#include <memory>
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

class DtccSolar
{

public:
    DtccSolar();
    DtccSolar(fArray2D vertices, iArray2D faces);
    DtccSolar(fArray2D vertices, iArray2D faces, std::vector<bool> face_mask, fArray2D sunSkyRays, fArray1D solidAngles);
    DtccSolar(fArray2D vertices, iArray2D faces, std::vector<bool> face_mask, fArray2D skyRays, fArray1D skySolidAngles, fArray2D sunRays, fArray1D sunSolidAngles);
    virtual ~DtccSolar();

    void CreateGeom(fArray2D vertices, iArray2D faces);
    void CreateGeomPlane();
    void CalcFaceMidPoints();
    void CalcFaceNormals();

    fArray1D GetRuntime();

    iArray2D GetMeshFaces();
    fArray2D GetMeshVertices();
    fArray2D GetFaceNormals();

    MatrixXfRM GetVPMatrix();
    VectorXf GetVPMatrixFlat();
    VectorXf GetIrradianceVector();
    VectorXf GetIrradianceMatrixFlat();

    MatrixXfRM GetVPMatrixSky();
    MatrixXfRM GetVPMatrixSun();
    VectorXf GetVPMatrixSkyFlat();
    VectorXf GetVPMatrixSunFlat();
    VectorXf GetIrradianceVectorSun();
    VectorXf GetIrradianceVectorSky();

    MatrixXfRM &GetIrradianceMatrix();
    MatrixXfRM &GetIrradianceMatrixSky();
    MatrixXfRM &GetIrradianceMatrixSun();
    VectorXf GetIrradianceMatrixSkyFlat();
    VectorXf GetIrradianceMatrixSunFlat();

    bool CalcVPMatrix(Rays *rays, MatrixXfRM &visProjMatrix, fArray2D &surfaceNormals);
    bool CalcIrradiance2Phase(Rays *rays, fArray1D &skySunVector, const MatrixXfRM &VP, VectorXf &irrVector);
    bool CalcIrradiance2Phase(Rays *rays, const MatrixXfRM &skySunMatrix, const MatrixXfRM &visProjMatrix, MatrixXfRM &irrMatrix);
    bool CalcIrradiance3Phase(Rays *skyRays, Rays *sunRays, VectorXf &skyS, VectorXf &sunS, const MatrixXfRM &skyVP, const MatrixXfRM &sunVP, VectorXf &skyIrrVec, VectorXf &sunIrrVec);
    bool CalcIrradiance3Phase(Rays *skyRays, Rays *sunRays, MatrixXfRM &skyS, MatrixXfRM &sunS, MatrixXfRM &skyVPMatrix, MatrixXfRM &sunVPMatrix, MatrixXfRM &skyIrrMatrix, MatrixXfRM &sunIrrMatrix);

    bool Run2PhaseAnalysis(fArray1D sunSkyVector);
    bool Run2PhaseAnalysis(fArray2D sunSkyMatrix);
    bool Run3PhaseAnalysis(fArray1D skyVector, fArray1D sunVector);
    bool Run3PhaseAnalysis(fArray2D skyMatrix, fArray2D sunMatrix);

private:
    std::unique_ptr<Accel> mAccel;

    Parameters mPp; // plane parameters

    int mVertexCount;
    int mFaceCount;

    float mRayTracingTime;
    float mMultiTime;
    float mEigenTime;
    float mTotalTime;

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

    VectorXf mIrrVector;
    VectorXf mIrrVectorSky;
    VectorXf mIrrVectorSun;

    MatrixXfRM mVPMatrix;
    MatrixXfRM mVPMatrixSky;
    MatrixXfRM mVPMatrixSun;
    MatrixXfRM mIrrMatrix;
    MatrixXfRM mIrrMatrixSky;
    MatrixXfRM mIrrMatrixSun;

    float mDomeSolidAngle = 2 * M_PI; // Solid angle of the dome, 2 * pi steradians

    // Ray objects for analysis
    Rays *mSunSkyRays = nullptr;
    Rays *mSkyRays = nullptr;
    Rays *mSunRays = nullptr;
};
