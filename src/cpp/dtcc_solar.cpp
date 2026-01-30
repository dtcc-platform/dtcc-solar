#include "dtcc_solar.h"
#include <omp.h>

// ------------------------------------------------------------
// NOTE: This .cpp assumes you will update dtcc_solar.h accordingly.
// Main changes:
//   - Two meshes: analysis mesh (receivers) + shading mesh (occluders)
//   - No faceMask / mMaskCount logic
//   - VP / irradiance computed for analysis faces only
//   - BVH built from shading mesh triangles
// ------------------------------------------------------------

// -------------------------
// Constructors
// -------------------------

DtccSolar::DtccSolar(fArray2D vertices, iArray2D faces)
{
    info("Creating DtccSolar instance with single mesh geometry (analysis == shading).");
    set_log_level(INFO);

    CreateGeom(vertices, faces, vertices, faces);

    CalcFaceMidPoints(); // analysis midpoints
    CalcFaceNormals();   // analysis normals

    info("Model setup with mesh geometry complete.");
}

// Two-phase: analysis mesh + shading mesh + combined (sun+sky) rays
DtccSolar::DtccSolar(
    fArray2D analysisVertices,
    iArray2D analysisFaces,
    fArray2D shadingVertices,
    iArray2D shadingFaces,
    fArray2D sunSkyRays,
    fArray1D solidAngles)
{
    info("-----------------------------------------------------");
    info("Creating DtccSolar instance with analysis + shading meshes.");
    set_log_level(INFO);

    Eigen::setNbThreads(std::thread::hardware_concurrency());
    info("Eigen using " + str(Eigen::nbThreads()) + " threads.");

    CreateGeom(analysisVertices, analysisFaces, shadingVertices, shadingFaces);

    CalcFaceMidPoints(); // analysis midpoints
    CalcFaceNormals();   // analysis normals

    mSunSkyRays = new Rays(sunSkyRays, solidAngles);

    info("Model setup complete.");
    info("-----------------------------------------------------");
}

// Three-phase: analysis mesh + shading mesh + separate sky + sun rays
DtccSolar::DtccSolar(
    fArray2D analysisVertices,
    iArray2D analysisFaces,
    fArray2D shadingVertices,
    iArray2D shadingFaces,
    fArray2D skyRays,
    fArray1D skySolidAngles,
    fArray2D sunRays,
    fArray1D sunSolidAngles)
{
    info("-----------------------------------------------------");
    info("Creating DtccSolar instance with analysis + shading meshes.");
    set_log_level(INFO);

    Eigen::setNbThreads(std::thread::hardware_concurrency());
    info("Eigen using " + str(Eigen::nbThreads()) + " threads.");

    CreateGeom(analysisVertices, analysisFaces, shadingVertices, shadingFaces);

    CalcFaceMidPoints(); // analysis midpoints
    CalcFaceNormals();   // analysis normals

    mSkyRays = new Rays(skyRays, skySolidAngles);
    mSunRays = new Rays(sunRays, sunSolidAngles);

    info("Model setup complete.");
    info("-----------------------------------------------------");
}

// -------------------------
// Destructor
// -------------------------

DtccSolar::~DtccSolar()
{
    // Delete rays if allocated
    if (mSunSkyRays)
    {
        delete mSunSkyRays;
        mSunSkyRays = nullptr;
    }
    if (mSkyRays)
    {
        delete mSkyRays;
        mSkyRays = nullptr;
    }
    if (mSunRays)
    {
        delete mSunRays;
        mSunRays = nullptr;
    }

    // Analysis arrays
    if (mFaceMidPts)
    {
        delete[] mFaceMidPts;
        mFaceMidPts = nullptr;
    }
    if (mFaceNormals)
    {
        delete[] mFaceNormals;
        mFaceNormals = nullptr;
    }
    if (mAnalysisVertices)
    {
        delete[] mAnalysisVertices;
        mAnalysisVertices = nullptr;
    }
    if (mAnalysisFaces)
    {
        delete[] mAnalysisFaces;
        mAnalysisFaces = nullptr;
    }

    // Shading arrays
    if (mShadingVertices)
    {
        delete[] mShadingVertices;
        mShadingVertices = nullptr;
    }
    if (mShadingFaces)
    {
        delete[] mShadingFaces;
        mShadingFaces = nullptr;
    }

    // Accel uses RAII
    mAccel.reset();
}

// -------------------------
// Getters
// -------------------------

fArray1D DtccSolar::GetRuntime()
{
    return {mRayTracingTime, mMultiTime, mTotalTime};
}

// Default mesh getters return ANALYSIS mesh (receivers)
iArray2D DtccSolar::GetMeshFaces()
{
    auto out = std::vector<std::vector<int>>(mAnalysisFaceCount, std::vector<int>(3, 0));
    for (int i = 0; i < mAnalysisFaceCount; i++)
    {
        Face f = mAnalysisFaces[i];
        out[i][0] = f.v0;
        out[i][1] = f.v1;
        out[i][2] = f.v2;
    }
    return out;
}

fArray2D DtccSolar::GetMeshVertices()
{
    auto out = std::vector<std::vector<float>>(mAnalysisVertexCount, std::vector<float>(3, 0.0f));
    for (int i = 0; i < mAnalysisVertexCount; i++)
    {
        Vertex v = mAnalysisVertices[i];
        out[i][0] = v.x;
        out[i][1] = v.y;
        out[i][2] = v.z;
    }
    return out;
}

fArray2D DtccSolar::GetFaceNormals()
{
    auto out = std::vector<std::vector<float>>(mAnalysisFaceCount, std::vector<float>(3, 0.0f));
    for (int i = 0; i < mAnalysisFaceCount; i++)
    {
        Vector v = mFaceNormals[i];
        out[i][0] = v.x;
        out[i][1] = v.y;
        out[i][2] = v.z;
    }
    return out;
}

MatrixXfRM DtccSolar::GetVPMatrix() { return mVPMatrix; }
MatrixXfRM DtccSolar::GetVPMatrixSky() { return mVPMatrixSky; }
MatrixXfRM DtccSolar::GetVPMatrixSun() { return mVPMatrixSun; }

MatrixXfRM &DtccSolar::GetIrradianceMatrix() { return mIrrMatrix; }
MatrixXfRM &DtccSolar::GetIrradianceMatrixSky() { return mIrrMatrixSky; }
MatrixXfRM &DtccSolar::GetIrradianceMatrixSun() { return mIrrMatrixSun; }

VectorXf DtccSolar::GetIrradianceMatrixFlat() { return RowSums(mIrrMatrix); }
VectorXf DtccSolar::GetIrradianceMatrixSkyFlat() { return RowSums(mIrrMatrixSky); }
VectorXf DtccSolar::GetIrradianceMatrixSunFlat() { return RowSums(mIrrMatrixSun); }

VectorXf DtccSolar::GetIrradianceVector() { return mIrrVector; }
VectorXf DtccSolar::GetIrradianceVectorSun() { return mIrrVectorSun; }
VectorXf DtccSolar::GetIrradianceVectorSky() { return mIrrVectorSky; }

VectorXf DtccSolar::GetSunHours() { return mSunHours; }
VectorXf DtccSolar::GetSkyViewFactor() { return mSkyViewFactor; }

// -------------------------
// Geometry creation
// -------------------------

static inline void FillVertices(Vertex *dst, const fArray2D &src)
{
    for (size_t i = 0; i < src.size(); i++)
    {
        if (src[i].size() != 3)
            error("Invalid vertex size.");
        dst[i].x = src[i][0];
        dst[i].y = src[i][1];
        dst[i].z = src[i][2];
    }
}

static inline void FillFaces(Face *dst, const iArray2D &src)
{
    for (size_t i = 0; i < src.size(); i++)
    {
        if (src[i].size() != 3)
            error("Invalid face size.");
        dst[i].v0 = src[i][0];
        dst[i].v1 = src[i][1];
        dst[i].v2 = src[i][2];
    }
}

static inline std::vector<Tri> BuildTrisFromMesh(const Vertex *V, int Vn, const Face *F, int Fn)
{
    std::vector<Tri> tris;
    tris.reserve(static_cast<size_t>(Fn));
    (void)Vn; // not strictly required here, but kept for sanity

    for (int i = 0; i < Fn; i++)
    {
        const Face &f = F[i];
        tris.emplace_back(
            Vec3(V[f.v0].x, V[f.v0].y, V[f.v0].z),
            Vec3(V[f.v1].x, V[f.v1].y, V[f.v1].z),
            Vec3(V[f.v2].x, V[f.v2].y, V[f.v2].z));
    }
    return tris;
}

void DtccSolar::CreateGeom(fArray2D analysisVertices, iArray2D analysisFaces,
                           fArray2D shadingVertices, iArray2D shadingFaces)
{
    // Store ANALYSIS mesh
    mAnalysisVertexCount = static_cast<int>(analysisVertices.size());
    mAnalysisFaceCount = static_cast<int>(analysisFaces.size());

    mAnalysisVertices = new Vertex[mAnalysisVertexCount];
    mAnalysisFaces = new Face[mAnalysisFaceCount];
    mFaceNormals = new Vector[mAnalysisFaceCount];
    mFaceMidPts = new Vertex[mAnalysisFaceCount];

    FillVertices(mAnalysisVertices, analysisVertices);
    FillFaces(mAnalysisFaces, analysisFaces);

    // Store SHADING mesh
    mShadingVertexCount = static_cast<int>(shadingVertices.size());
    mShadingFaceCount = static_cast<int>(shadingFaces.size());

    mShadingVertices = new Vertex[mShadingVertexCount];
    mShadingFaces = new Face[mShadingFaceCount];

    FillVertices(mShadingVertices, shadingVertices);
    FillFaces(mShadingFaces, shadingFaces);

    // Build BVH from SHADING mesh (occluders)
    std::vector<Tri> tris = BuildTrisFromMesh(mShadingVertices, mShadingVertexCount, mShadingFaces, mShadingFaceCount);
    mAccel = std::make_unique<Accel>(tris, "high");

    info("Analysis mesh: vertices=" + str(mAnalysisVertexCount) + ", faces=" + str(mAnalysisFaceCount));
    info("Shading mesh: vertices=" + str(mShadingVertexCount) + ", faces=" + str(mShadingFaceCount));
    info("BVH built with " + str(mShadingFaceCount) + " shading triangles.");
}

// -------------------------
// Analysis midpoints + normals (computed for ANALYSIS mesh only)
// -------------------------

void DtccSolar::CalcFaceMidPoints()
{
    // mFaceMidPts allocated in CreateGeom/CreateGeomPlane with size mAnalysisFaceCount
    for (int i = 0; i < mAnalysisFaceCount; i++)
    {
        Face f = mAnalysisFaces[i];

        float x = mAnalysisVertices[f.v0].x + mAnalysisVertices[f.v1].x + mAnalysisVertices[f.v2].x;
        float y = mAnalysisVertices[f.v0].y + mAnalysisVertices[f.v1].y + mAnalysisVertices[f.v2].y;
        float z = mAnalysisVertices[f.v0].z + mAnalysisVertices[f.v1].z + mAnalysisVertices[f.v2].z;

        Vertex v;
        v.x = x / 3.0f;
        v.y = y / 3.0f;
        v.z = z / 3.0f;

        mFaceMidPts[i] = v;
    }
}

void DtccSolar::CalcFaceNormals()
{
    // Normals for analysis faces (receiver orientation)
    for (int i = 0; i < mAnalysisFaceCount; i++)
    {
        Face f = mAnalysisFaces[i];
        Vector v1 = CreateVector(mAnalysisVertices[f.v0], mAnalysisVertices[f.v1]);
        Vector v2 = CreateVector(mAnalysisVertices[f.v0], mAnalysisVertices[f.v2]);

        v1 = UnitizeVector(v1);
        v2 = UnitizeVector(v2);

        Vector n = CrossProduct(v1, v2);
        n = UnitizeVector(n);

        mFaceNormals[i] = n;
    }
}

// -------------------------
// Irradiance
// -------------------------

bool DtccSolar::CalcIrradiance2Phase(Rays *rays, fArray1D &skySunVector, const MatrixXfRM &VP, VectorXf &E)
{
    if (!rays)
    {
        error("Rays not initialized.");
        return false;
    }

    const int rayCount = rays->GetRayCount();
    if (static_cast<int>(skySunVector.size()) != rayCount)
    {
        error("Sky-sun vector length does not match ray count.");
        return false;
    }

    if (VP.rows() != mAnalysisFaceCount || VP.cols() != rayCount)
    {
        error("VP dimensions do not match (analysisFaceCount x rayCount).");
        return false;
    }

    const Eigen::VectorXf S = VectorToEigen(skySunVector);

    if (E.size() != mAnalysisFaceCount)
        E.resize(mAnalysisFaceCount);

    auto start = hrClock::now();
    E.noalias() = VP * S;
    auto end = hrClock::now();

    info("Irradiance vector min: " + std::to_string(E.minCoeff()) + ", max: " + std::to_string(E.maxCoeff()));

    fDuration duration = end - start;
    mMultiTime = duration.count();
    info("Irradiance calculation with Eigen completed in " + str(duration.count()) + " seconds.");
    return true;
}

bool DtccSolar::CalcIrradiance2Phase(Rays *rays, const MatrixXfRM &skySun, const MatrixXfRM &visProj, MatrixXfRM &irradiance)
{
    if (!rays)
    {
        error("Rays is not initialized. Cannot calculate irradiance.");
        return false;
    }

    const Eigen::Index faceCount = visProj.rows();
    const Eigen::Index rayCountVP = visProj.cols();

    const Eigen::Index rayCountSS = skySun.rows();
    const Eigen::Index timeSteps = skySun.cols();

    info("Vis-Proj-Matrix has shape: (" + str(static_cast<size_t>(faceCount)) + ", " + str(static_cast<size_t>(rayCountVP)) + ")");
    info("Sky-Sun-Matrix has shape: (" + str(static_cast<size_t>(rayCountSS)) + ", " + str(static_cast<size_t>(timeSteps)) + ")");
    info("Irradiance matrix shape: (" + str(static_cast<size_t>(faceCount)) + ", " + str(static_cast<size_t>(timeSteps)) + ")");

    const int rayCountRays = rays->GetRayCount();

    if (rayCountSS != rayCountRays)
    {
        error("Matrix shape mismatch. skySun rows do not match ray count.");
        return false;
    }
    if (rayCountVP != rayCountSS)
    {
        error("Matrix shape mismatch. visProj cols do not match skySun rows.");
        return false;
    }
    if (faceCount != mAnalysisFaceCount)
    {
        error("Matrix shape mismatch. visProj rows do not match analysisFaceCount.");
        return false;
    }

    irradiance.resize(mAnalysisFaceCount, static_cast<int>(timeSteps));

    auto start = hrClock::now();
    irradiance.noalias() = visProj * skySun;
    auto end = hrClock::now();
    fDuration duration = end - start;

    mMultiTime = duration.count();
    info("Irradiance calculation with Eigen completed in " + str(duration.count()) + " seconds.");
    return true;
}

bool DtccSolar::CalcIrradiance3Phase(Rays *skyRays, Rays *sunRays,
                                     VectorXf &skyS, VectorXf &sunS,
                                     const MatrixXfRM &skyVP, const MatrixXfRM &sunVP,
                                     VectorXf &Esky, VectorXf &Esun)
{
    if (!skyRays || !sunRays)
    {
        error("Rays are not initialized. Cannot calculate irradiance.");
        return false;
    }

    const int skyRayCount = skyRays->GetRayCount();
    const int sunRayCount = sunRays->GetRayCount();

    info("Sky-VP has shape: (" + str(static_cast<size_t>(skyVP.rows())) + ", " + str(static_cast<size_t>(skyVP.cols())) + ")");
    info("Sun-VP has shape: (" + str(static_cast<size_t>(sunVP.rows())) + ", " + str(static_cast<size_t>(sunVP.cols())) + ")");
    info("Sky vector length: (" + str(static_cast<int>(skyS.size())) + ")");
    info("Sun vector length: (" + str(static_cast<int>(sunS.size())) + ")");

    if (skyVP.rows() != mAnalysisFaceCount || sunVP.rows() != mAnalysisFaceCount)
    {
        error("VP matrix row count does not match analysisFaceCount.");
        return false;
    }
    if (skyVP.cols() != skyRayCount)
    {
        error("Sky VP matrix col count does not match sky ray count.");
        return false;
    }
    if (sunVP.cols() != sunRayCount)
    {
        error("Sun VP matrix col count does not match sun ray count.");
        return false;
    }
    if (static_cast<int>(skyS.size()) != skyRayCount)
    {
        error("Sky vector length does not match sky ray count.");
        return false;
    }
    if (static_cast<int>(sunS.size()) != sunRayCount)
    {
        error("Sun vector length does not match sun ray count.");
        return false;
    }

    Esky.resize(mAnalysisFaceCount);
    Esun.resize(mAnalysisFaceCount);

    auto start = hrClock::now();
    Esky.noalias() = skyVP * skyS;
    Esun.noalias() = sunVP * sunS;
    auto end = hrClock::now();

    fDuration duration = end - start;
    mMultiTime = duration.count();
    info("3-phase irradiance vector calculation completed in " + str(duration.count()) + " seconds.");

    return true;
}

bool DtccSolar::CalcIrradiance3Phase(Rays *skyRays, Rays *sunRays,
                                     MatrixXfRM &skyS, MatrixXfRM &sunS,
                                     MatrixXfRM &skyVP, MatrixXfRM &sunVP,
                                     MatrixXfRM &skyE, MatrixXfRM &sunE)
{
    auto skyMatShape = GetShape(skyS);
    auto sunMatShape = GetShape(sunS);

    auto skyVPShape = GetShape(skyVP);
    auto sunVPShape = GetShape(sunVP);

    auto skyEShape = GetShape(skyE);
    auto sunEShape = GetShape(sunE);

    info("Sky-Matrix has shape: (" + str(skyMatShape.first) + ", " + str(skyMatShape.second) + ")");
    info("Sky-Vis-Proj-Matrix has shape: (" + str(skyVPShape.first) + ", " + str(skyVPShape.second) + ")");
    info("Sky-Irradiance-Matrix has shape: (" + str(skyEShape.first) + ", " + str(skyEShape.second) + ")");

    info("Sun-Matrix has shape: (" + str(sunMatShape.first) + ", " + str(sunMatShape.second) + ")");
    info("Sun-Vis-Proj-Matrix has shape: (" + str(sunVPShape.first) + ", " + str(sunVPShape.second) + ")");
    info("Sun-Irradiance-Matrix has shape: (" + str(sunEShape.first) + ", " + str(sunEShape.second) + ")");

    const int skyTimeSteps = skyMatShape.second;

    if (skyMatShape.first != skyRays->GetRayCount() || sunMatShape.first != sunRays->GetRayCount())
    {
        error("Matrix shape mismatch. Cannot calculate irradiance.");
        return false;
    }

    skyE.resize(mAnalysisFaceCount, skyTimeSteps);
    auto start1 = hrClock::now();
    skyE.noalias() = skyVP * skyS;
    auto end1 = hrClock::now();
    fDuration duration1 = end1 - start1;
    info("Irradiance from sky calculated with Eigen in " + str(duration1.count()) + " seconds.");

    sunE.resize(mAnalysisFaceCount, skyTimeSteps);
    VectorXf diagSun = sunS.diagonal();
    auto start2 = hrClock::now();
    sunE.noalias() = sunVP * diagSun.asDiagonal();
    auto end2 = hrClock::now();
    fDuration duration2 = end2 - start2;
    info("Irradiance from sun calculated with Eigen in " + str(duration2.count()) + " seconds.");

    mMultiTime = duration1.count() + duration2.count();
    return true;
}

// -------------------------
// VP matrix (computed for ANALYSIS faces, occlusion against SHADING BVH)
// -------------------------

bool DtccSolar::CalcVPMatrix(Rays *rays,
                             MatrixXfRM &visProj,
                             fArray2D &surfaceNormals,
                             bool computeSunHours,
                             bool computeSkyViewFactor)
{
    if (!rays)
    {
        error("Rays is not initialized. Cannot compute visibility-projection matrix.");
        return false;
    }
    if (!mAccel)
    {
        error("BVH accel not initialized. Cannot ray trace.");
        return false;
    }

    const fArray2D rayDirs = rays->GetRayDirections();   // (nRays x 3)
    const fArray1D solidAngles = rays->GetSolidAngles(); // (nRays)
    const int nRays = rays->GetRayCount();

    if (static_cast<int>(surfaceNormals.size()) != mAnalysisFaceCount)
    {
        error("surfaceNormals size does not match analysisFaceCount.");
        return false;
    }

    // Resize VP (analysisFaces x rays)
    if (visProj.rows() != mAnalysisFaceCount || visProj.cols() != nRays)
        visProj.resize(mAnalysisFaceCount, nRays);
    visProj.setZero();

    if (computeSkyViewFactor)
    {
        mSkyViewFactor.resize(mAnalysisFaceCount);
        mSkyViewFactor.setZero();
    }
    if (computeSunHours)
    {
        mSunHours.resize(mAnalysisFaceCount);
        mSunHours.setZero();
    }

    int hitCounter = 0;
    int hitAttempts = 0;

    auto start = hrClock::now();
    info("Calculating VP matrix with BVH for " + str(mAnalysisFaceCount) +
         " analysis faces and " + str(nRays) + " rays. (Occluders: " + str(mShadingFaceCount) + " faces)");

    static constexpr size_t stack_size = 64;

#pragma omp parallel for schedule(dynamic) reduction(+ : hitCounter, hitAttempts)
    for (int i = 0; i < mAnalysisFaceCount; ++i)
    {
        const auto &n = surfaceNormals[i];
        Vec3 face_origin(mFaceMidPts[i].x, mFaceMidPts[i].y, mFaceMidPts[i].z);

        float denom = 0.0f; // sum over front hemisphere: (n·d) Ω
        float numer = 0.0f; // sum over visible rays:   (n·d) Ω
        int sunHits = 0;

        float *row = visProj.data() + static_cast<size_t>(i) * static_cast<size_t>(nRays);

        for (int j = 0; j < nRays; ++j)
        {
            const auto &r = rayDirs[j];

            const float dot = n[0] * r[0] + n[1] * r[1] + n[2] * r[2];
            if (dot <= 0.0f)
                continue;

            const float w = dot * solidAngles[j];
            denom += w;

            // Copy ray template (your Rays class should return BVH Ray-compatible objects here)
            Ray ray = rays->GetRays()[j];
            ray.org = face_origin;

            bool occluded = false;
            bvh::v2::SmallStack<Bvh::Index, stack_size> stack;

            mAccel->bvh.intersect<false, false>(
                ray, mAccel->bvh.get_root().index, stack,
                [&](size_t begin, size_t end)
                {
                    for (size_t k = begin; k < end; ++k)
                    {
                        if (mAccel->precomputed_tris[k].intersect(ray))
                        {
                            occluded = true;
                            return true; // early exit
                        }
                    }
                    return false;
                });

            hitAttempts++;

            if (occluded)
            {
                hitCounter++;
            }
            else
            {
                row[j] = w;
                numer += w;
                if (computeSunHours)
                    sunHits += 1;
            }
        }

        if (computeSkyViewFactor)
            mSkyViewFactor(i) = (denom > 0.0f) ? (numer / denom) : 0.0f;

        if (computeSunHours)
            mSunHours(i) = static_cast<float>(sunHits);
    }

    auto end = hrClock::now();
    fDuration duration = end - start;
    mRayTracingTime = duration.count();

    info("VP matrix calculated successfully.");
    info("Found " + str(hitCounter) + " intersections in " + str(hitAttempts) + " attempts");
    info("Time elapsed: " + str(duration.count()) + " seconds.");

    return true;
}

// -------------------------
// Run analysis (unchanged in spirit, but uses analysisFaceCount everywhere)
// -------------------------

bool DtccSolar::Run2PhaseAnalysis(fArray1D sunSkyVec)
{
    info("-----------------------------------------------------");
    info("Running 2-phase 1D analysis: E = VP * S");
    auto start = hrClock::now();

    if (!mSunSkyRays)
    {
        error("mSunSkyRays is not initialized.");
        return false;
    }

    VectorXf E;
    fArray2D surfaceNormals = GetFaceNormals();

    MatrixXfRM VP;
    if (!CalcVPMatrix(mSunSkyRays, VP, surfaceNormals, false, true))
        return false;

    if (!CalcIrradiance2Phase(mSunSkyRays, sunSkyVec, VP, E))
        return false;

    mVPMatrix = std::move(VP);
    mIrrVector = std::move(E);

    auto end = hrClock::now();
    fDuration duration = end - start;
    mTotalTime = duration.count();

    info("2-phase analysis completed successfully.");
    info("-----------------------------------------------------");
    return true;
}

bool DtccSolar::Run2PhaseAnalysis(fArray2D sunSkyMat)
{
    info("-----------------------------------------------------");
    info("Running 2-phase 2D analysis: E = VP * S");
    auto start = hrClock::now();

    if (!mSunSkyRays)
    {
        error("mSunSkyRays is not initialized.");
        return false;
    }

    const int numRays = mSunSkyRays->GetRayCount();
    if (sunSkyMat.empty() || sunSkyMat[0].empty())
    {
        error("sunSkyMat is empty. Cannot run analysis.");
        return false;
    }
    if (static_cast<int>(sunSkyMat.size()) != numRays)
    {
        error("sunSkyMat row count does not match ray count.");
        return false;
    }

    fArray2D surfaceNormals = GetFaceNormals();

    MatrixXfRM VP;
    if (!CalcVPMatrix(mSunSkyRays, VP, surfaceNormals, false, true))
        return false;

    MatrixXfRM skySun = VectorToEigen(sunSkyMat);
    MatrixXfRM E;

    if (!CalcIrradiance2Phase(mSunSkyRays, skySun, VP, E))
        return false;

    mVPMatrix = std::move(VP);
    mIrrMatrix = std::move(E);

    auto end = hrClock::now();
    fDuration duration = end - start;
    mTotalTime = duration.count();

    info("2-phase analysis completed successfully.");
    info("-----------------------------------------------------");
    return true;
}

bool DtccSolar::Run3PhaseAnalysis(fArray1D skyVector, fArray1D sunVector)
{
    info("-----------------------------------------------------");
    info("Running 3-phase 1D analysis: E = VP_sky * S_sky + VP_sun * S_sun");
    auto start = hrClock::now();

    if (!mSkyRays || !mSunRays)
    {
        error("mSkyRays or mSunRays is not initialized.");
        return false;
    }

    MatrixXfRM skyVP;
    MatrixXfRM sunVP;

    VectorXf Esky, Esun;

    fArray2D surfaceNormals = GetFaceNormals();
    VectorXf skyS = VectorToEigen(skyVector);
    VectorXf sunS = VectorToEigen(sunVector);

    if (!CalcVPMatrix(mSkyRays, skyVP, surfaceNormals, false, true))
        return false;

    if (!CalcVPMatrix(mSunRays, sunVP, surfaceNormals, true, false))
        return false;

    if (!CalcIrradiance3Phase(mSkyRays, mSunRays, skyS, sunS, skyVP, sunVP, Esky, Esun))
        return false;

    mVPMatrixSky = std::move(skyVP);
    mVPMatrixSun = std::move(sunVP);
    mIrrVectorSky = std::move(Esky);
    mIrrVectorSun = std::move(Esun);

    auto end = hrClock::now();
    fDuration duration = end - start;
    mTotalTime = duration.count();

    info("3-phase analysis completed successfully.");
    info("-----------------------------------------------------");
    return true;
}

bool DtccSolar::Run3PhaseAnalysis(fArray2D skyMatrix, fArray2D sunMatrix)
{
    info("-----------------------------------------------------");
    info("Running 3-phase 2D analysis: E = VP_sky * S_sky + VP_sun * S_sun");
    auto start = hrClock::now();

    if (!mSkyRays || !mSunRays)
    {
        error("mSkyRays or mSunRays is not initialized.");
        return false;
    }

    const int numSkyRays = mSkyRays->GetRayCount();
    const int numSunRays = mSunRays->GetRayCount();

    if (skyMatrix.empty() || skyMatrix[0].empty())
    {
        error("skyMatrix is empty. Cannot run 3-phase analysis.");
        return false;
    }
    if (sunMatrix.empty() || sunMatrix[0].empty())
    {
        error("sunMatrix is empty. Cannot run 3-phase analysis.");
        return false;
    }
    if (static_cast<int>(skyMatrix.size()) != numSkyRays)
    {
        error("skyMatrix row count does not match number of sky rays.");
        return false;
    }
    if (static_cast<int>(sunMatrix.size()) != numSunRays)
    {
        error("sunMatrix row count does not match number of sun rays.");
        return false;
    }

    fArray2D surfaceNormals = GetFaceNormals();

    MatrixXfRM skyS = VectorToEigen(skyMatrix);
    MatrixXfRM sunS = VectorToEigen(sunMatrix);

    MatrixXfRM skyVP;
    MatrixXfRM sunVP;

    if (!CalcVPMatrix(mSkyRays, skyVP, surfaceNormals, false, true))
        return false;

    if (!CalcVPMatrix(mSunRays, sunVP, surfaceNormals, true, false))
        return false;

    MatrixXfRM skyE, sunE;
    if (!CalcIrradiance3Phase(mSkyRays, mSunRays, skyS, sunS, skyVP, sunVP, skyE, sunE))
        return false;

    mVPMatrixSky = std::move(skyVP);
    mVPMatrixSun = std::move(sunVP);
    mIrrMatrixSky = std::move(skyE);
    mIrrMatrixSun = std::move(sunE);

    auto end = hrClock::now();
    fDuration duration = end - start;
    mTotalTime = duration.count();

    info("3-phase analysis completed successfully.");
    info("-----------------------------------------------------");
    return true;
}

#ifdef PYTHON_MODULE

namespace py = pybind11;

// Create a 2D NumPy view of a RowMajor Eigen matrix without copying.
template <class Mat>
py::array arr_from_eigen_rm(Mat &M, py::handle owner)
{
    static_assert(Mat::IsRowMajor, "Matrix must be RowMajor for this helper.");
    const py::ssize_t rows = static_cast<py::ssize_t>(M.rows());
    const py::ssize_t cols = static_cast<py::ssize_t>(M.cols());
    using Scalar = typename Mat::Scalar;
    const py::ssize_t stride_row = static_cast<py::ssize_t>(sizeof(Scalar) * cols);
    const py::ssize_t stride_col = static_cast<py::ssize_t>(sizeof(Scalar));
    return py::array(py::dtype::of<Scalar>(), {rows, cols}, {stride_row, stride_col}, M.data(), owner);
}

// Copy Eigen::VectorXf -> numpy 1D
inline py::array vec_to_numpy_1d(Eigen::VectorXf v)
{
    const py::ssize_t n = static_cast<py::ssize_t>(v.size());
    py::array_t<float> out(n);
    std::copy(v.data(), v.data() + v.size(), out.mutable_data());
    py::buffer_info info(out.mutable_data(), sizeof(float), py::format_descriptor<float>::format(), 1,
                         {n}, {static_cast<py::ssize_t>(sizeof(float))});
    return py::array(info, out);
}

PYBIND11_MODULE(py_solar, m)
{
    py::class_<DtccSolar>(m, "PySolar")
        // -----------------------------
        // Option C: 2-phase constructor
        // analysis mesh + optional shading mesh + (sun+sky) rays
        // -----------------------------
        .def(py::init([](std::vector<std::vector<float>> analysisVertices,
                         std::vector<std::vector<int>> analysisFaces,
                         std::vector<std::vector<float>> shadingVertices,
                         std::vector<std::vector<int>> shadingFaces,
                         std::vector<std::vector<float>> sunSkyRays,
                         std::vector<float> solidAngles)
                      { return new DtccSolar(
                            std::move(analysisVertices),
                            std::move(analysisFaces),
                            std::move(shadingVertices),
                            std::move(shadingFaces),
                            std::move(sunSkyRays),
                            std::move(solidAngles)); }),
             py::arg("analysis_vertices"),
             py::arg("analysis_faces"),
             py::arg("shading_vertices") = std::vector<std::vector<float>>{}, // empty => use analysis mesh
             py::arg("shading_faces") = std::vector<std::vector<int>>{},      // empty => use analysis mesh
             py::arg("sun_sky_rays"),
             py::arg("solid_angles"))

        // -----------------------------
        // Option C: 3-phase constructor
        // analysis mesh + optional shading mesh + sky rays + sun rays
        // -----------------------------
        .def(py::init([](std::vector<std::vector<float>> analysisVertices,
                         std::vector<std::vector<int>> analysisFaces,
                         std::vector<std::vector<float>> shadingVertices,
                         std::vector<std::vector<int>> shadingFaces,
                         std::vector<std::vector<float>> skyRays,
                         std::vector<float> skySolidAngles,
                         std::vector<std::vector<float>> sunRays,
                         std::vector<float> sunSolidAngles)
                      { return new DtccSolar(
                            std::move(analysisVertices),
                            std::move(analysisFaces),
                            std::move(shadingVertices),
                            std::move(shadingFaces),
                            std::move(skyRays),
                            std::move(skySolidAngles),
                            std::move(sunRays),
                            std::move(sunSolidAngles)); }),
             py::arg("analysis_vertices"),
             py::arg("analysis_faces"),
             py::arg("shading_vertices") = std::vector<std::vector<float>>{}, // empty => use analysis mesh
             py::arg("shading_faces") = std::vector<std::vector<int>>{},      // empty => use analysis mesh
             py::arg("sky_rays"),
             py::arg("sky_solid_angles"),
             py::arg("sun_rays"),
             py::arg("sun_solid_angles"))

        // Existing methods
        .def("get_mesh_faces", [](DtccSolar &self)
             { return py::cast(self.GetMeshFaces()); })
        .def("get_mesh_vertices", [](DtccSolar &self)
             { return py::cast(self.GetMeshVertices()); })
        .def("get_face_normals", [](DtccSolar &self)
             { return py::cast(self.GetFaceNormals()); })
        .def("run_2_phase_analysis_vec", [](DtccSolar &self, std::vector<float> sun_sky_vec)
             { return py::cast(self.Run2PhaseAnalysis(sun_sky_vec)); })
        .def("run_2_phase_analysis_mat", [](DtccSolar &self, std::vector<std::vector<float>> sun_sky_mat)
             { return py::cast(self.Run2PhaseAnalysis(sun_sky_mat)); })
        .def("run_3_phase_analysis_vec", [](DtccSolar &self, std::vector<float> sky_vec, std::vector<float> sun_vec)
             { return py::cast(self.Run3PhaseAnalysis(sky_vec, sun_vec)); })
        .def("run_3_phase_analysis_mat", [](DtccSolar &self, std::vector<std::vector<float>> sky_mat, std::vector<std::vector<float>> sun_mat)
             { return py::cast(self.Run3PhaseAnalysis(sky_mat, sun_mat)); })
        .def("get_irradiance_vector", [](DtccSolar &self)
             { return vec_to_numpy_1d(self.GetIrradianceVector()); })
        .def("get_irradiance_vector_sun", [](DtccSolar &self)
             { return vec_to_numpy_1d(self.GetIrradianceVectorSun()); })
        .def("get_irradiance_vector_sky", [](DtccSolar &self)
             { return vec_to_numpy_1d(self.GetIrradianceVectorSky()); })
        .def("get_irradiance_matrix", [](DtccSolar &self)
             { return arr_from_eigen_rm(self.GetIrradianceMatrix(), py::cast(&self)); }, py::return_value_policy::reference_internal)
        .def("get_irradiance_matrix_flat", [](DtccSolar &self)
             { return vec_to_numpy_1d(self.GetIrradianceMatrixFlat()); })
        .def("get_irradiance_matrix_sky", [](DtccSolar &self)
             { return arr_from_eigen_rm(self.GetIrradianceMatrixSky(), py::cast(&self)); }, py::return_value_policy::reference_internal)
        .def("get_irradiance_matrix_sky_flat", [](DtccSolar &self)
             { return vec_to_numpy_1d(self.GetIrradianceMatrixSkyFlat()); })
        .def("get_irradiance_matrix_sun", [](DtccSolar &self)
             { return arr_from_eigen_rm(self.GetIrradianceMatrixSun(), py::cast(&self)); }, py::return_value_policy::reference_internal)
        .def("get_irradiance_matrix_sun_flat", [](DtccSolar &self)
             { return vec_to_numpy_1d(self.GetIrradianceMatrixSunFlat()); })
        .def("get_runtime", [](DtccSolar &self)
             { return py::cast(self.GetRuntime()); })
        .def("get_sun_hours", [](DtccSolar &self)
             { return vec_to_numpy_1d(self.GetSunHours()); })
        .def("get_sky_view_factor", [](DtccSolar &self)
             { return vec_to_numpy_1d(self.GetSkyViewFactor()); });
}
#endif
