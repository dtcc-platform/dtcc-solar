#include "dtcc_solar.h"
#include <omp.h>

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

    // Allocate result arrays
    mSunVisibleRayCount = VectorXf::Zero(mAnalysisFaceCount);
    mSkyViewFactor = VectorXf::Zero(mAnalysisFaceCount);
    mIrrVector = VectorXf::Zero(mAnalysisFaceCount);
    mIrrVectorSky = VectorXf::Zero(mAnalysisFaceCount);
    mIrrVectorSun = VectorXf::Zero(mAnalysisFaceCount);

    mCombinedRays = new Rays(skyRays, skySolidAngles);
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
    if (mCombinedRays)
    {
        delete mCombinedRays;
        mCombinedRays = nullptr;
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

VectorXf DtccSolar::GetSunVisibleRays() { return mSunVisibleRayCount; }
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

void DtccSolar::CreateGeom(fArray2D analysisVertices, iArray2D analysisFaces, fArray2D shadingVertices, iArray2D shadingFaces)
{
    // Store ANALYSIS mesh (receivers)
    mAnalysisVertexCount = static_cast<int>(analysisVertices.size());
    mAnalysisFaceCount = static_cast<int>(analysisFaces.size());

    mAnalysisVertices = new Vertex[mAnalysisVertexCount];
    mAnalysisFaces = new Face[mAnalysisFaceCount];
    mFaceNormals = new Vector[mAnalysisFaceCount];
    mFaceMidPts = new Vertex[mAnalysisFaceCount];

    FillVertices(mAnalysisVertices, analysisVertices);
    FillFaces(mAnalysisFaces, analysisFaces);

    // Store SHADING mesh (optional extra occluders)
    const bool hasShading = !(shadingVertices.empty() || shadingFaces.empty());
    if (!hasShading)
        info("No shading mesh provided -> occluders will be analysis mesh only (self-shading enabled).");

    mShadingVertexCount = hasShading ? static_cast<int>(shadingVertices.size()) : 0;
    mShadingFaceCount = hasShading ? static_cast<int>(shadingFaces.size()) : 0;

    if (hasShading)
    {
        mShadingVertices = new Vertex[mShadingVertexCount];
        mShadingFaces = new Face[mShadingFaceCount];

        FillVertices(mShadingVertices, shadingVertices);
        FillFaces(mShadingFaces, shadingFaces);
    }
    else
    {
        // Keep pointers null if no shading mesh
        mShadingVertices = nullptr;
        mShadingFaces = nullptr;
    }

    // ------------------------------------------------------------
    // Build BVH occluders from (shading mesh + analysis mesh)
    // This enables self-shading of analysis mesh even when a separate
    // shading mesh is provided.
    // ------------------------------------------------------------
    std::vector<Tri> tris;

    // Reserve roughly to avoid reallocations
    const int totalFaces = mAnalysisFaceCount + (hasShading ? mShadingFaceCount : 0);
    tris.reserve(static_cast<size_t>(totalFaces));

    // 1) Add SHADING tris (if provided)
    if (hasShading && mShadingFaceCount > 0)
    {
        std::vector<Tri> shadeTris = BuildTrisFromMesh(mShadingVertices, mShadingVertexCount, mShadingFaces, mShadingFaceCount);
        tris.insert(tris.end(), shadeTris.begin(), shadeTris.end());
    }

    // >>> SET OFFSET HERE <<<
    mAnalysisTriOffsetInBVH = static_cast<int>(tris.size());

    // 2) Add ANALYSIS tris (always) to enable self-occlusion
    std::vector<Tri> analysisTris = BuildTrisFromMesh(mAnalysisVertices, mAnalysisVertexCount, mAnalysisFaces, mAnalysisFaceCount);
    tris.insert(tris.end(), analysisTris.begin(), analysisTris.end());

    // Build BVH from combined occluders
    mAccel = std::make_unique<Accel>(tris, "high");

    info("Analysis mesh: vertices=" + str(mAnalysisVertexCount) + ", faces=" + str(mAnalysisFaceCount));
    if (hasShading)
        info("Shading mesh: vertices=" + str(mShadingVertexCount) + ", faces=" + str(mShadingFaceCount));
    else
        info("Shading mesh: (none)");

    info("BVH built with " + str(static_cast<int>(tris.size())) + " occluder triangles (shading + analysis).");
}

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

bool DtccSolar::CalcIrradiance2Phase(Rays *rays, VectorXf &S, const MatrixXfRM &VP, VectorXf &E)
{
    if (!rays)
    {
        error("Rays not initialized.");
        return false;
    }

    const int rayCount = rays->GetRayCount();
    if (S.size() != rayCount)
    {
        error("Sky-sun vector length does not match ray count.");
        return false;
    }

    if (VP.rows() != mAnalysisFaceCount || VP.cols() != rayCount)
    {
        error("VP dimensions do not match (analysisFaceCount x rayCount).");
        return false;
    }

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

bool DtccSolar::CalcIrradiance5Phase(Rays *skyRays, Rays *sunRays, VectorXf &skyS, VectorXf &sunS, const MatrixXfRM &skyVP, const MatrixXfRM &sunVP, VectorXf &Esky, VectorXf &Esun)
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
    info("5-phase irradiance vector calculation completed in " + str(duration.count()) + " seconds.");

    return true;
}

bool DtccSolar::CalcIrradiance5Phase(Rays *skyRays, Rays *sunRays, MatrixXfRM &skyS, MatrixXfRM &sunS, MatrixXfRM &skyVP, MatrixXfRM &sunVP, MatrixXfRM &skyE, MatrixXfRM &sunE)
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
    auto start2 = hrClock::now();
    sunE.noalias() = sunVP * sunS; // (N x k) * (k x T)
    auto end2 = hrClock::now();
    fDuration duration2 = end2 - start2;
    info("Irradiance from sun calculated with Eigen in " + str(duration2.count()) + " seconds.");

    mMultiTime = duration1.count() + duration2.count();
    return true;
}

// -------------------------
// VP matrix (computed for ANALYSIS faces, occlusion against SHADING BVH)
// -------------------------
bool DtccSolar::CalcVPMatrix(Rays *rays, MatrixXfRM &visProj, fArray2D &surfaceNormals, bool computeSunHours, bool computeSVF, const std::vector<int> *sunHourWeights)
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

    const fArray2D &rayDirs = rays->GetRayDirections();   // (nRays x 3)
    const fArray1D &solidAngles = rays->GetSolidAngles(); // (nRays)
    const int nRays = rays->GetRayCount();

    if (static_cast<int>(surfaceNormals.size()) != mAnalysisFaceCount)
    {
        error("surfaceNormals size does not match analysisFaceCount.");
        return false;
    }
    if (static_cast<int>(rayDirs.size()) != nRays || static_cast<int>(solidAngles.size()) != nRays)
    {
        error("Rays internal arrays do not match GetRayCount().");
        return false;
    }

    if (computeSunHours && sunHourWeights)
    {
        if (static_cast<int>(sunHourWeights->size()) != nRays)
        {
            error("sunHourWeights size does not match nRays.");
            return false;
        }
    }

    // Resize VP (analysisFaces x rays)
    if (visProj.rows() != mAnalysisFaceCount || visProj.cols() != nRays)
        visProj.resize(mAnalysisFaceCount, nRays);
    visProj.setZero();

    if (computeSVF)
    {
        mSkyViewFactor.resize(mAnalysisFaceCount);
        mSkyViewFactor.setZero();
    }
    if (computeSunHours)
    {
        mSunVisibleRayCount.resize(mAnalysisFaceCount);
        mSunVisibleRayCount.setZero();
    }

    // ----------------------------------------------------------
    // SVF denominator for hemisphere dome: sum of all solid angles
    // (Your dome is already a hemisphere, so no horizon checks here)
    // ----------------------------------------------------------
    double svfDenom = 0.0;
    if (computeSVF)
    {
        for (int j = 0; j < nRays; ++j)
            svfDenom += static_cast<double>(solidAngles[static_cast<size_t>(j)]);

        if (svfDenom <= 0.0)
        {
            error("SkyViewFactor denominator is zero. Are your solid angles valid?");
            return false;
        }
    }

    // ----------------------------------------------------------
    // Combined BVH contains shading tris + analysis tris.
    // We must skip the receiver face’s own analysis triangle:
    // triIndexSelf = mAnalysisTriOffsetInBVH + i
    // ----------------------------------------------------------
    const size_t triCount = mAccel->precomputed_tris.size();

    // Sanity (optional but useful)
    if (mAnalysisTriOffsetInBVH + static_cast<size_t>(mAnalysisFaceCount) > triCount)
    {
        error("BVH does not contain expected analysis triangle range. Check CreateGeom BVH build.");
        return false;
    }

    int hitCounter = 0;
    int hitAttempts = 0;

    auto start = hrClock::now();
    info("Calculating VP matrix with BVH for " + str(mAnalysisFaceCount) +
         " analysis faces and " + str(nRays) + " rays. (Occluder tris: " + str(triCount) + ")");

    static constexpr size_t stack_size = 64;

#pragma omp parallel for schedule(dynamic) reduction(+ : hitCounter, hitAttempts)
    for (int i = 0; i < mAnalysisFaceCount; ++i)
    {
        const auto &n = surfaceNormals[static_cast<size_t>(i)];
        Vec3 face_origin(mFaceMidPts[i].x, mFaceMidPts[i].y, mFaceMidPts[i].z);

        double svfNumer = 0.0; // sum Ω over visible rays that are in front of the face
        int sunHits = 0;       // weighted (if weights provided) or plain count

        float *row = visProj.data() + static_cast<size_t>(i) * static_cast<size_t>(nRays);

        const size_t selfTriIndex = mAnalysisTriOffsetInBVH + static_cast<size_t>(i);

        for (int j = 0; j < nRays; ++j)
        {
            const auto &r = rayDirs[static_cast<size_t>(j)];

            // Front-side test w.r.t face normal
            const float dot = n[0] * r[0] + n[1] * r[1] + n[2] * r[2];
            if (dot <= 0.0f)
                continue;

            // VP weight (unchanged)
            const float w = dot * solidAngles[static_cast<size_t>(j)];

            // Copy ray template and set origin to face midpoint
            Ray ray = rays->GetRays()[static_cast<size_t>(j)];
            ray.org = face_origin;

            bool occluded = false;
            bvh::v2::SmallStack<Bvh::Index, stack_size> stack;

            mAccel->bvh.intersect<false, false>(
                ray, mAccel->bvh.get_root().index, stack,
                [&](size_t begin, size_t end)
                {
                    for (size_t k = begin; k < end; ++k)
                    {
                        // Skip self triangle (receiver face) to avoid self-occlusion
                        if (k == selfTriIndex)
                            continue;

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
                // Visible => store VP entry
                row[j] = w;

                // SVF: sum solid angles (not cosine-weighted)
                if (computeSVF)
                    svfNumer += static_cast<double>(solidAngles[static_cast<size_t>(j)]);

                // Sun hours / visible sun rays
                if (computeSunHours)
                {
                    const int add = sunHourWeights ? (*sunHourWeights)[static_cast<size_t>(j)] : 1;
                    sunHits += add;
                }
            }
        }

        if (computeSVF)
            mSkyViewFactor(i) = static_cast<float>(svfNumer / svfDenom);

        if (computeSunHours)
            mSunVisibleRayCount(i) = static_cast<float>(sunHits);
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
// Run analysis methods
// -------------------------

bool DtccSolar::RunAnalysis(fArray2D skyMatrix, fArray2D sunMatrix, iArray1D activeSunIndices, bool is1D, bool computeSunHours, bool computeSVF)
{
    auto start = hrClock::now();
    bool isSameShape = SameShape(skyMatrix, sunMatrix);
    bool success = false;

    // Build unique active patches + weights (geometry only, no DNI)
    std::vector<int> activeUnique;
    std::vector<int> weights;
    BuildActiveUniqueAndWeights(activeSunIndices, activeUnique, weights);

    // sum of weights for sanity check (should equal number of timesteps represented by activeSunIndices)
    int weightSum = 0;
    for (int w : weights)
        weightSum += w;
    info("Active sun patches: " + str(activeUnique.size()) + ", sum of weights: " + str(weightSum));

    if (is1D)
    {
        VectorXf skyVec = RowSumToVectorXf(skyMatrix);
        VectorXf sunVec = RowSumToVectorXf(sunMatrix);

        if (isSameShape && !computeSunHours)
        {
            VectorXf combinedVec = skyVec + sunVec;
            success = Run2PhaseAnalysis(combinedVec, computeSVF);
        }
        else
        {
            success = Run5PhaseAnalysis(skyVec, sunVec, activeUnique, weights, computeSunHours, computeSVF);
        }
    }
    else
    {
        MatrixXfRM skyMat = VectorToEigenRM(skyMatrix);
        MatrixXfRM sunMat = VectorToEigenRM(sunMatrix);

        if (isSameShape && !computeSunHours)
        {
            MatrixXfRM combinedMat = skyMat + sunMat;
            success = Run2PhaseAnalysis(combinedMat, computeSVF);
        }
        else
        {
            success = Run5PhaseAnalysis(skyMat, sunMat, activeUnique, weights, computeSunHours, computeSVF);
        }
    }

    auto end = hrClock::now();
    fDuration duration = end - start;
    mTotalTime = duration.count();

    return success;
}

bool DtccSolar::Run2PhaseAnalysis(VectorXf sunSkyVec, bool computeSkyViewFactor)
{
    info("-----------------------------------------------------");
    info("Running 2-phase 1D analysis: E = VP * S");

    if (!mCombinedRays)
    {
        error("mCombinedRays is not initialized.");
        return false;
    }

    VectorXf E;
    fArray2D surfaceNormals = GetFaceNormals();

    MatrixXfRM VP;
    if (!CalcVPMatrix(mCombinedRays, VP, surfaceNormals, false, computeSkyViewFactor, nullptr))
        return false;

    if (!CalcIrradiance2Phase(mCombinedRays, sunSkyVec, VP, E))
        return false;

    mVPMatrix = std::move(VP);
    mIrrVector = std::move(E);

    info("2-phase analysis completed successfully.");
    info("-----------------------------------------------------");
    return true;
}

bool DtccSolar::Run2PhaseAnalysis(MatrixXfRM sunSkyMat, bool computeSkyViewFactor)
{
    info("-----------------------------------------------------");
    info("Running 2-phase 2D analysis: E = VP * S");
    auto start = hrClock::now();

    if (!mCombinedRays)
    {
        error("mCombinedRays is not initialized.");
        return false;
    }

    const int numRays = mCombinedRays->GetRayCount();

    if (sunSkyMat.rows() != numRays)
    {
        error("sunSkyMat row count does not match ray count.");
        return false;
    }

    fArray2D surfaceNormals = GetFaceNormals();

    MatrixXfRM VP;
    if (!CalcVPMatrix(mCombinedRays, VP, surfaceNormals, false, computeSkyViewFactor, nullptr))
        return false;

    MatrixXfRM E;

    if (!CalcIrradiance2Phase(mCombinedRays, sunSkyMat, VP, E))
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

bool DtccSolar::Run5PhaseAnalysis(VectorXf skyS, VectorXf sunS, const iArray1D &activeSunIndices, const iArray1D &sunHourWeightsActive, bool computeSunHours, bool computeSkyViewFactor)
{
    info("-----------------------------------------------------");
    info("Running 5-phase 1D analysis: E = VP_sky * S_sky + VP_sun(active) * S_sun(active)");

    if (!mSkyRays || !mSunRays)
    {
        error("mSkyRays or mSunRays is not initialized.");
        return false;
    }

    const int P = mSkyRays->GetRayCount();
    const int K = mSunRays->GetRayCount();

    if (skyS.size() != P)
    {
        error("skyS size does not match number of sky rays.");
        return false;
    }
    if (sunS.size() != K)
    {
        error("sunS size does not match number of sun rays.");
        return false;
    }

    if (!activeSunIndices.empty() && static_cast<int>(sunHourWeightsActive.size()) != static_cast<int>(activeSunIndices.size()))
    {
        error("sunHourWeightsActive must have same length as activeSunIndices.");
        return false;
    }

    // Normals once
    fArray2D surfaceNormals = GetFaceNormals();

    // 1) VP for sky (full)
    MatrixXfRM skyVP;
    if (!CalcVPMatrix(mSkyRays, skyVP, surfaceNormals, false, computeSkyViewFactor, nullptr))
        return false;

    // 2) Filter sun rays + sun vector (active only)
    MatrixXfRM sunVP;
    VectorXf sunS_active;
    std::unique_ptr<Rays> sunRaysFiltered;

    if (!activeSunIndices.empty())
    {
        sunS_active = SelectRows(sunS, activeSunIndices);
        sunRaysFiltered = MakeFilteredRays(mSunRays, activeSunIndices);

        const std::vector<int> *weightsPtr = nullptr;
        if (computeSunHours)
            weightsPtr = &sunHourWeightsActive;

        if (!CalcVPMatrix(sunRaysFiltered.get(), sunVP, surfaceNormals, computeSunHours, false, weightsPtr))
            return false;
    }
    else
    {
        // No active sun patches => no sun contribution, no geometric sun hours
        sunVP.resize(mAnalysisFaceCount, 0);
        sunS_active.resize(0);
        if (computeSunHours)
        {
            mSunVisibleRayCount.resize(mAnalysisFaceCount);
            mSunVisibleRayCount.setZero();
        }
    }

    // 3) Irradiance vectors
    VectorXf Esky(mAnalysisFaceCount);
    Esky.noalias() = skyVP * skyS;

    VectorXf Esun(mAnalysisFaceCount);
    Esun.setZero();
    if (sunVP.cols() > 0)
        Esun.noalias() = sunVP * sunS_active;

    // 4) Store
    mVPMatrixSky = std::move(skyVP);
    mVPMatrixSun = std::move(sunVP);

    mIrrVectorSky = std::move(Esky);
    mIrrVectorSun = std::move(Esun);
    mIrrVector = mIrrVectorSky + mIrrVectorSun;

    info("5-phase analysis completed successfully.");
    info("-----------------------------------------------------");
    return true;
}

bool DtccSolar::Run5PhaseAnalysis(MatrixXfRM skyS, MatrixXfRM sunS, const iArray1D &activeSunIndices, const iArray1D &sunHourWeightsActive, bool computeSunHours, bool computeSkyViewFactor)
{
    info("-----------------------------------------------------");
    info("Running 5-phase 2D analysis: E = VP_sky * S_sky + VP_sun(active) * S_sun(active)");

    if (!mSkyRays || !mSunRays)
    {
        error("mSkyRays or mSunRays is not initialized.");
        return false;
    }

    const int P = mSkyRays->GetRayCount();
    const int K = mSunRays->GetRayCount();

    if (skyS.rows() != P)
    {
        error("skyMatrix row count does not match number of sky rays.");
        return false;
    }
    if (sunS.rows() != K)
    {
        error("sunMatrix row count does not match number of sun rays.");
        return false;
    }
    if (skyS.cols() != sunS.cols())
    {
        error("Time-step mismatch: skyMatrix.cols() != sunMatrix.cols().");
        return false;
    }

    if (!activeSunIndices.empty() &&
        static_cast<int>(sunHourWeightsActive.size()) != static_cast<int>(activeSunIndices.size()))
    {
        error("sunHourWeightsActive must have same length as activeSunIndices.");
        return false;
    }

    const int T = static_cast<int>(skyS.cols());
    fArray2D surfaceNormals = GetFaceNormals();

    // 1) VP for sky (full)
    MatrixXfRM skyVP;
    if (!CalcVPMatrix(mSkyRays, skyVP, surfaceNormals,
                      /*computeSunHours=*/false,
                      /*computeSkyViewFactor=*/computeSkyViewFactor,
                      /*sunHourWeights=*/nullptr))
        return false;

    // 2) Filter sun rays + sun matrix (active only)
    MatrixXfRM sunVP;
    MatrixXfRM sunS_active;                // (k x T)
    std::unique_ptr<Rays> sunRaysFiltered; // k rays

    if (!activeSunIndices.empty())
    {
        sunS_active = SelectRows(sunS, activeSunIndices); // (k x T)
        sunRaysFiltered = MakeFilteredRays(mSunRays, activeSunIndices);

        const std::vector<int> *weightsPtr = nullptr;
        if (computeSunHours)
            weightsPtr = &sunHourWeightsActive;

        if (!CalcVPMatrix(sunRaysFiltered.get(), sunVP, surfaceNormals,
                          /*computeSunHours=*/computeSunHours,
                          /*computeSkyViewFactor=*/false,
                          /*sunHourWeights=*/weightsPtr))
            return false;
    }
    else
    {
        // No active sun patches => no sun contribution
        sunVP.resize(mAnalysisFaceCount, 0);
        sunS_active.resize(0, T);

        if (computeSunHours)
        {
            mSunVisibleRayCount.resize(mAnalysisFaceCount);
            mSunVisibleRayCount.setZero();
        }
    }

    // 3) Irradiance matrices
    MatrixXfRM skyE(mAnalysisFaceCount, T);
    skyE.noalias() = skyVP * skyS;

    MatrixXfRM sunE(mAnalysisFaceCount, T);
    sunE.setZero();
    if (sunVP.cols() > 0)
        sunE.noalias() = sunVP * sunS_active; // (N x k) * (k x T)

    // 4) Store
    mVPMatrixSky = std::move(skyVP);
    mVPMatrixSun = std::move(sunVP);
    mIrrMatrixSky = std::move(skyE);
    mIrrMatrixSun = std::move(sunE);

    mIrrMatrix = mIrrMatrixSky + mIrrMatrixSun;

    info("5-phase analysis completed successfully.");
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
        .def("analyse", [](DtccSolar &self, std::vector<std::vector<float>> sky_mat, std::vector<std::vector<float>> sun_mat, std::vector<int> activeSunIndices, bool is1D, bool computeSunHours, bool computeSkyViewFactor)
             { return py::cast(self.RunAnalysis(sky_mat, sun_mat, activeSunIndices, is1D, computeSunHours, computeSkyViewFactor)); })
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
        .def("get_sun_visible_rays", [](DtccSolar &self)
             { return vec_to_numpy_1d(self.GetSunVisibleRays()); })
        .def("get_sky_view_factor", [](DtccSolar &self)
             { return vec_to_numpy_1d(self.GetSkyViewFactor()); });
}
#endif
