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
#include <unordered_map>

#include "common.h"
#include "rays.h"
#include "logging.h"
#include "rays_utils.h"

#ifdef PYTHON_MODULE
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

class DtccSolar
{
public:
    // Plane constructor: analysis and shading are the same generated plane
    DtccSolar();

    // Single-mesh constructor: analysis == shading (backwards-ish compatibility)
    DtccSolar(fArray2D vertices, iArray2D faces);

    // Two-mesh constructors: analysis mesh + shading mesh
    DtccSolar(
        fArray2D analysisVertices, iArray2D analysisFaces,
        fArray2D shadingVertices, iArray2D shadingFaces,
        fArray2D sunSkyRays, fArray1D solidAngles);

    DtccSolar(
        fArray2D analysisVertices, iArray2D analysisFaces,
        fArray2D shadingVertices, iArray2D shadingFaces,
        fArray2D skyRays, fArray1D skySolidAngles,
        fArray2D sunRays, fArray1D sunSolidAngles);

    virtual ~DtccSolar();

    // Geometry setup
    void CreateGeomPlane();                                       // creates both analysis+shading as same plane
    void CreateGeomSingleMesh(fArray2D vertices, iArray2D faces); // analysis==shading
    void CreateGeom(
        fArray2D analysisVertices, iArray2D analysisFaces,
        fArray2D shadingVertices, iArray2D shadingFaces);

    void CalcFaceMidPoints(); // for analysis faces
    void CalcFaceNormals();   // for analysis faces

    // Runtime
    fArray1D GetRuntime();

    std::string GetAnalysisLog();

    // Mesh getters (analysis)
    iArray2D GetMeshFaces();    // analysis faces
    fArray2D GetMeshVertices(); // analysis vertices
    fArray2D GetFaceNormals();  // analysis normals

    // Optional: shading mesh getters
    iArray2D GetShadingMeshFaces();
    fArray2D GetShadingMeshVertices();

    // Results
    MatrixXfRM GetVPMatrix();
    VectorXf GetVPMatrixFlat(); // if you implement as RowSums(mVPMatrix) or similar

    VectorXf GetIrradianceVector();
    VectorXf GetIrradianceMatrixFlat();

    VectorXf GetSunVisibleRays();
    VectorXf GetSkyViewFactor();

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

    bool CalcVPMatrix(Rays *rays, MatrixXfRM &visProj, fArray2D &surfaceNormals, bool computeSunHours, bool computeSkyViewFactor, const std::vector<int> *sunHourWeights);

    bool CalcIrradiance2Phase(Rays *rays, VectorXf &skySunVector, const MatrixXfRM &VP, VectorXf &irrVector);
    bool CalcIrradiance2Phase(Rays *rays, const MatrixXfRM &skySunMatrix, const MatrixXfRM &visProjMatrix, MatrixXfRM &irrMatrix);

    bool CalcIrradiance5Phase(Rays *skyRays, Rays *sunRays, VectorXf &skyS, VectorXf &sunS, const MatrixXfRM &skyVP, const MatrixXfRM &sunVP, VectorXf &skyIrrVec, VectorXf &sunIrrVec);
    bool CalcIrradiance5Phase(Rays *skyRays, Rays *sunRays, MatrixXfRM &skyS, MatrixXfRM &sunS, MatrixXfRM &skyVPMatrix, MatrixXfRM &sunVPMatrix, MatrixXfRM &skyIrrMatrix, MatrixXfRM &sunIrrMatrix);

    // Analyses
    bool Run2PhaseAnalysis(VectorXf sunSkyVector, bool computeSkyViewFactor);
    bool Run2PhaseAnalysis(MatrixXfRM sunSkyMatrix, bool computeSkyViewFactor);
    bool Run5PhaseAnalysis(VectorXf skyS, VectorXf sunS, const iArray1D &activeSunIndices, const iArray1D &sunHourWeightsActive, bool computeSunHours, bool computeSkyViewFactor);
    bool Run5PhaseAnalysis(MatrixXfRM skyMatrix, MatrixXfRM sunMatrix, const iArray1D &activeSunIndices, const iArray1D &sunHourWeightsActive, bool computeSunHours, bool computeSkyViewFactor);

    bool RunAnalysis(fArray2D skyMatrix, fArray2D sunMatrix, iArray1D activeSunIndices, bool is1D, bool computeSunHours, bool computeSkyViewFactor);

private:
    // BVH built from shading mesh
    std::unique_ptr<Accel> mAccel;

    Parameters mPp; // plane parameters

    std::string mAnalysisLog;

    // Analysis mesh counts
    int mAnalysisVertexCount = 0;
    int mAnalysisFaceCount = 0;
    int mAnalysisTriOffsetInBVH = 0;

    // Shading mesh counts
    int mShadingVertexCount = 0;
    int mShadingFaceCount = 0;

    // Timings
    float mRayTracingTime = 0.0f;
    float mMultiTime = 0.0f;
    float mEigenTime = 0.0f;
    float mTotalTime = 0.0f;

    // Analysis mesh storage
    Face *mAnalysisFaces = nullptr;
    Vertex *mAnalysisVertices = nullptr;
    Vertex *mFaceMidPts = nullptr;  // analysis face midpoints
    Vector *mFaceNormals = nullptr; // analysis face normals

    // Shading mesh storage (ray-occluders)
    Face *mShadingFaces = nullptr;
    Vertex *mShadingVertices = nullptr;

    // Results (all sized by analysis faces)
    VectorXf mSkyViewFactor;
    VectorXf mSunVisibleRayCount;

    VectorXf mIrrVector;
    VectorXf mIrrVectorSky;
    VectorXf mIrrVectorSun;

    MatrixXfRM mVPMatrix;
    MatrixXfRM mVPMatrixSky;
    MatrixXfRM mVPMatrixSun;

    MatrixXfRM mIrrMatrix;
    MatrixXfRM mIrrMatrixSky;
    MatrixXfRM mIrrMatrixSun;

    float mDomeSolidAngle = 2 * M_PI; // hemisphere solid angle

    // Ray objects for analysis
    Rays *mCombinedRays = nullptr;
    Rays *mSkyRays = nullptr;
    Rays *mSunRays = nullptr;
};
