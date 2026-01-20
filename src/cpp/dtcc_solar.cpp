#include "dtcc_solar.h"
#include <omp.h>

DtccSolar::DtccSolar()
{
    info("Creating DtccSolar instance with default constructor.");

    set_log_level(INFO);

    // Plane parameters
    mPp.xMin = -10.0f;
    mPp.xMax = 10.0f;
    mPp.yMin = -10.0f;
    mPp.yMax = 10.0f;
    mPp.xPadding = 0.0f;
    mPp.yPadding = 0.0f;
    mPp.xCount = 201;
    mPp.yCount = 201;

    mVertexCount = mPp.xCount * mPp.yCount;
    mFaceCount = (mPp.xCount - 1) * (mPp.yCount - 1) * 2;

    info("Model setup with plane geometry contains:");
    info("Number of vertices: " + str(mVertexCount));
    info("Number of faces: " + str(mFaceCount));

    mMaskCount = mFaceCount;
    mApplyMask = false;
    mFaceMask = std::vector<bool>(mFaceCount, true);
    mFaceNormals = new Vector[mFaceCount];

    CreateGeomPlane();
    CalcFaceMidPoints();
    CalcFaceNormals();

    info("Model setup with plane geometry complete.");
}

DtccSolar::DtccSolar(fArray2D vertices, iArray2D faces)
{
    info("Creating DtccSolar instance with mesh geometry.");

    set_log_level(INFO);

    mVertexCount = (int)vertices.size();
    mFaceCount = (int)faces.size();
    mFaceNormals = new Vector[mFaceCount];

    mMaskCount = mFaceCount;
    mApplyMask = false;
    mFaceMask = std::vector<bool>(mFaceCount, true);

    CreateGeom(vertices, faces);
    CalcFaceMidPoints();
    CalcFaceNormals();

    info("Model setup with mesh geometry complete.");
}

DtccSolar::DtccSolar(fArray2D vertices, iArray2D faces, bArray1D faceMask, fArray2D sunSkyRays, fArray1D solidAngles)
{
    info("-----------------------------------------------------");
    info("Creating DtccSolar instance with mesh geometry.");
    set_log_level(INFO);

    Eigen::setNbThreads(std::thread::hardware_concurrency());
    info("Eigen using " + str(Eigen::nbThreads()) + " threads.");

    mVertexCount = (int)vertices.size();
    mFaceCount = (int)faces.size();
    mFaceNormals = new Vector[mFaceCount];

    mApplyMask = false;
    mMaskCount = 0;
    mFaceMask = faceMask;
    mMaskCount = 0;
    for (int i = 0; i < mFaceCount; i++)
        if (mFaceMask[i])
            mMaskCount++;

    // Print mask count
    info("Mask count: " + str(mMaskCount));

    CreateGeom(vertices, faces);
    CalcFaceMidPoints();
    CalcFaceNormals();

    mSunSkyRays = new Rays(sunSkyRays, solidAngles);

    info("Model setup with mesh geometry complete.");
    info("-----------------------------------------------------");
}

DtccSolar::DtccSolar(fArray2D vertices, iArray2D faces, bArray1D faceMask, fArray2D skyRays, fArray1D skySolidAngles, fArray2D sunRays, fArray1D sunSolidAngles)
{
    info("-----------------------------------------------------");
    info("Creating DtccSolar instance with mesh geometry.");
    set_log_level(INFO);

    Eigen::setNbThreads(std::thread::hardware_concurrency());
    info("Eigen using " + str(Eigen::nbThreads()) + " threads.");

    mVertexCount = (int)vertices.size();
    mFaceCount = (int)faces.size();
    mFaceNormals = new Vector[mFaceCount];

    mApplyMask = false;
    mMaskCount = 0;
    mFaceMask = faceMask;
    for (int i = 0; i < mFaceCount; i++)
        if (mFaceMask[i])
            mMaskCount++;

    CreateGeom(vertices, faces);
    CalcFaceMidPoints();
    CalcFaceNormals();

    mSkyRays = new Rays(skyRays, skySolidAngles);
    mSunRays = new Rays(sunRays, sunSolidAngles);

    info("Model setup with mesh geometry complete.");
    info("-----------------------------------------------------");
}

DtccSolar::~DtccSolar()
{
    // Delete rays if allocated
    if (mSunSkyRays != nullptr)
    {
        delete mSunSkyRays;
        mSunSkyRays = nullptr;
    }

    if (mSkyRays != nullptr)
    {
        delete mSkyRays;
        mSkyRays = nullptr;
    }

    if (mSunRays != nullptr)
    {
        delete mSunRays;
        mSunRays = nullptr;
    }

    // Delete geometry arrays
    if (mFaceMidPts != nullptr)
    {
        delete[] mFaceMidPts;
        mFaceMidPts = nullptr;
    }

    if (mFaceNormals != nullptr)
    {
        delete[] mFaceNormals;
        mFaceNormals = nullptr;
    }

    if (mVertices != nullptr)
    {
        delete[] mVertices;
        mVertices = nullptr;
    }

    if (mFaces != nullptr)
    {
        delete[] mFaces;
        mFaces = nullptr;
    }

    // Accel uses RAII, automatic cleanup via unique_ptr
    mAccel.reset();
}

iArray2D DtccSolar::GetMeshFaces()
{
    auto faces = std::vector<std::vector<int>>(mFaceCount, std::vector<int>(3, 0));

    for (int i = 0; i < mFaceCount; i++)
    {
        Face f = mFaces[i];
        faces[i][0] = f.v0;
        faces[i][1] = f.v1;
        faces[i][2] = f.v2;
    }

    return faces;
}

fArray2D DtccSolar::GetMeshVertices()
{
    auto vertices = std::vector<std::vector<float>>(mVertexCount, std::vector<float>(3, 0));

    for (int i = 0; i < mVertexCount; i++)
    {
        Vertex v = mVertices[i];
        vertices[i][0] = v.x;
        vertices[i][1] = v.y;
        vertices[i][2] = v.z;
    }

    return vertices;
}

fArray2D DtccSolar::GetFaceNormals()
{
    auto vertices = std::vector<std::vector<float>>(mFaceCount, std::vector<float>(3, 0));

    for (int i = 0; i < mFaceCount; i++)
    {
        Vector v = mFaceNormals[i];
        vertices[i][0] = v.x;
        vertices[i][1] = v.y;
        vertices[i][2] = v.z;
    }

    return vertices;
}

// Combined results

fArray2D DtccSolar::GetVisibilityMatrixTot()
{
    return mVisMatrixTot;
}

fArray2D DtccSolar::GetProjectionMatrixTot()
{
    return mProjMatrixTot;
}

fArray2D DtccSolar::GetIrradianceMatrixTot()
{
    return mIrrMatrixTot;
}

fArray1D DtccSolar::GetVisibilityVectorTot()
{
    return Flatten2D(mVisMatrixTot);
}

fArray1D DtccSolar::GetProjectionVectorTot()
{
    return Flatten2D(mProjMatrixTot);
}

fArray1D DtccSolar::GetIrradianceVectorTot()
{
    return Flatten2D(mIrrMatrixTot);
}

// Sky results

fArray2D DtccSolar::GetVisibilityMatrixSky()
{
    return mVisMatrixSky;
}

fArray2D DtccSolar::GetProjectionMatrixSky()
{
    return mProjMatrixSky;
}

fArray2D DtccSolar::GetIrradianceMatrixSky()
{
    return mIrrMatrixSky;
}

fArray1D DtccSolar::GetVisibilityVectorSky()
{
    return Flatten2D(mVisMatrixSky);
}

fArray1D DtccSolar::GetProjectionVectorSky()
{
    return Flatten2D(mProjMatrixSky);
}

fArray1D DtccSolar::GetIrradianceVectorSky()
{
    return Flatten2D(mIrrMatrixSky);
}

// Sun results

fArray2D DtccSolar::GetVisibilityMatrixSun()
{
    return mVisMatrixSun;
}

fArray2D DtccSolar::GetProjectionMatrixSun()
{
    return mProjMatrixSun;
}

fArray2D DtccSolar::GetIrradianceMatrixSun()
{
    return mIrrMatrixSun;
}

fArray1D DtccSolar::GetVisibilityVectorSun()
{
    return Flatten2D(mVisMatrixSun);
}

fArray1D DtccSolar::GetProjectionVectorSun()
{
    return Flatten2D(mProjMatrixSun);
}

fArray1D DtccSolar::GetIrradianceVectorSun()
{
    return Flatten2D(mIrrMatrixSun);
}

fArray1D DtccSolar::Flatten2D(fArray2D &mat)
{
    // Collapse a (m x t) matrix into a (m x 1) vector by summing over t
    if (mat.empty())
        return {};

    const size_t rows = mat.size();
    const size_t cols = mat[0].size();

    fArray1D flat;
    flat.reserve(rows);

    for (const auto &row : mat)
    {
        if (row.size() != cols)
        {
            throw std::runtime_error("Flatten2D: ragged rows detected");
        }

        float sum = 0.0f;
        for (float v : row)
        {
            sum += v;
        }
        flat.push_back(sum);
    }

    return flat;
}

void DtccSolar::CreateGeom(fArray2D vertices, iArray2D faces)
{
    // Store vertices locally
    mVertices = new Vertex[mVertexCount];
    mFaces = new Face[mFaceCount];

    for (long unsigned int i = 0; i < vertices.size(); i++)
    {
        if (vertices[i].size() == 3)
        {
            Vertex &v = mVertices[i];
            v.x = vertices[i][0];
            v.y = vertices[i][1];
            v.z = vertices[i][2];
        }
        else
            error("Invalid vertex size in DtccSolar::CreateGeom.");
    }

    for (long unsigned int i = 0; i < faces.size(); i++)
    {
        if (faces[i].size() == 3)
        {
            Face &f = mFaces[i];
            f.v0 = faces[i][0];
            f.v1 = faces[i][1];
            f.v2 = faces[i][2];
        }
        else
            error("Invalid face size in DtccSolar::CreateGeom.");
    }

    // Build triangles for BVH
    std::vector<Tri> tris;
    tris.reserve(mFaceCount);
    for (int i = 0; i < mFaceCount; i++)
    {
        Face &f = mFaces[i];
        tris.emplace_back(
            Vec3(mVertices[f.v0].x, mVertices[f.v0].y, mVertices[f.v0].z),
            Vec3(mVertices[f.v1].x, mVertices[f.v1].y, mVertices[f.v1].z),
            Vec3(mVertices[f.v2].x, mVertices[f.v2].y, mVertices[f.v2].z));
    }

    // Build BVH using Accel class
    mAccel = std::make_unique<Accel>(tris, "high");

    info("BVH built with " + str(mFaceCount) + " triangles.");
}

void DtccSolar::CreateGeomPlane()
{
    /* create triangle mesh */
    const float xStep = (mPp.xMax - mPp.xMin) / (mPp.xCount - 1);
    const float yStep = (mPp.yMax - mPp.yMin) / (mPp.yCount - 1);

    const int nVertices = mPp.xCount * mPp.yCount;
    const int nFaces = (mPp.xCount - 1) * (mPp.yCount - 1) * 2;

    mVertices = new Vertex[nVertices];
    mFaces = new Face[nFaces];

    /* create plane mesh */
    int face_index = 0;
    for (int i = 0; i < mPp.yCount; i++)
    {
        float y = mPp.yMin + i * yStep;
        for (int j = 0; j < mPp.xCount; j++)
        {
            float x = mPp.xMin + j * xStep;

            Vertex &v = mVertices[i * mPp.xCount + j];
            v.x = x;
            v.y = y;
            v.z = 0.0f;

            if (i > 0 && j > 0)
            {
                // Add two triangles
                int base_index = j + (mPp.xCount * i);
                mFaces[face_index].v0 = base_index - mPp.xCount - 1;
                mFaces[face_index].v1 = base_index - mPp.xCount;
                mFaces[face_index].v2 = base_index;

                mFaces[face_index + 1].v0 = base_index - mPp.xCount - 1;
                mFaces[face_index + 1].v1 = base_index;
                mFaces[face_index + 1].v2 = base_index - 1;

                face_index += 2;
            }
        }
    }

    // Build triangles for BVH
    std::vector<Tri> tris;
    tris.reserve(nFaces);
    for (int i = 0; i < nFaces; i++)
    {
        Face &f = mFaces[i];
        tris.emplace_back(
            Vec3(mVertices[f.v0].x, mVertices[f.v0].y, mVertices[f.v0].z),
            Vec3(mVertices[f.v1].x, mVertices[f.v1].y, mVertices[f.v1].z),
            Vec3(mVertices[f.v2].x, mVertices[f.v2].y, mVertices[f.v2].z));
    }

    // Build BVH using Accel class
    mAccel = std::make_unique<Accel>(tris, "high");

    info("BVH built with " + str(nFaces) + " triangles (plane geometry).");
}

void DtccSolar::CalcFaceMidPoints()
{
    mFaceMidPts = new Vertex[mFaceCount];

    // Calculate face mid pts
    for (int i = 0; i < mFaceCount; i++)
    {
        Face f = mFaces[i];
        float x = mVertices[f.v0].x + mVertices[f.v1].x + mVertices[f.v2].x;
        float y = mVertices[f.v0].y + mVertices[f.v1].y + mVertices[f.v2].y;
        float z = mVertices[f.v0].z + mVertices[f.v1].z + mVertices[f.v2].z;

        Vertex v;
        v.x = x / 3.0f;
        v.y = y / 3.0f;
        v.z = z / 3.0f;

        mFaceMidPts[i] = v;
    }
}

void DtccSolar::CalcFaceNormals()
{
    // Normals are pointing upwards for counter clockwise winding of vertices
    for (int i = 0; i < mFaceCount; i++)
    {
        Face f = mFaces[i];
        Vector v1 = CreateVector(mVertices[f.v0], mVertices[f.v1]);
        Vector v2 = CreateVector(mVertices[f.v0], mVertices[f.v2]);

        v1 = UnitizeVector(v1);
        v2 = UnitizeVector(v2);

        Vector vNormal = CrossProduct(v1, v2);
        vNormal = UnitizeVector(vNormal);

        mFaceNormals[i] = vNormal;
    }
}

bool DtccSolar::CalcProjMatrix(Rays *rays, fArray2D &mProjectionMatrix)
{
    if (!rays)
    {
        error("RayDome is not initialized. Cannot compute projection matrix.");
        return false;
    }

    fArray2D surfaceNormals = GetFaceNormals();
    fArray2D rayDirections = rays->GetRayDirections();
    fArray1D raySolidAngles = rays->GetSolidAngles();
    size_t numRays = rayDirections.size();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < mFaceCount; ++i)
    {
        auto n = surfaceNormals[i];
        if (mFaceMask[i])
        {
            for (size_t j = 0; j < numRays; ++j)
            {
                auto r = rayDirections[j];
                float dot = n[0] * r[0] + n[1] * r[1] + n[2] * r[2];
                mProjectionMatrix[i][j] = std::max(0.0f, dot) * raySolidAngles[j];
            }
        }
    }

    info("Projection matrix was calculated successfully.");

    return true;
}

bool DtccSolar::CalcVisMatrix(Rays *rays, fArray2D &visMatrix)
{
    int hitCounter = 0;
    int hitAttempts = 0;
    float hitPortion = 0.0f;
    float thisPortion = 0.0f;

    mSkyViewFactor = fArray1D(mFaceCount, 0);
    auto start = hrClock::now();
    info("Calculating visibility matrix with BVH for " + str(mMaskCount) + " faces and " + str(rays->GetRayCount()) + " rays.");

    static constexpr size_t stack_size = 64;
    #pragma omp parallel for schedule(dynamic) reduction(+:hitCounter,hitAttempts)
    for (int i = 0; i < mFaceCount; i++)
    {
        if (mFaceMask[i])
        {
            rays->TranslateRays(mFaceMidPts[i]);
            int nRays = rays->GetRayCount();
            hitPortion = 0.0;

            for (int j = 0; j < nRays; j++)
            {
                Ray ray = rays->GetRays()[j];

                // Check for any intersection (occlusion test) using BVH traversal
                bool occluded = false;
                bvh::v2::SmallStack<Bvh::Index, stack_size> stack;

                mAccel->bvh.intersect<false, false>(ray, mAccel->bvh.get_root().index, stack,
                                                    [&](size_t begin, size_t end)
                                                    {
                                                        for (size_t k = begin; k < end; ++k)
                                                        {
                                                            if (auto hit = mAccel->precomputed_tris[k].intersect(ray))
                                                            {
                                                                occluded = true;
                                                                return true; // Early exit on first hit
                                                            }
                                                        }
                                                        return false;
                                                    });

                if (occluded)
                {
                    hitCounter++;
                    thisPortion = rays->GetSolidAngles()[j] / mDomeSolidAngle;
                    hitPortion = hitPortion + thisPortion;
                    visMatrix[i][j] = 0.0f;
                }
                hitAttempts++;
            }
            mSkyViewFactor[i] = 1.0 - hitPortion;
            if (i > 0 && i % 10000 == 0)
                info("Raytracing for " + str(i) + " faces completed.");
        }
    }

    info("Visibility matrix calculated successfully");
    info("Found " + str(hitCounter) + " intersections in " + str(hitAttempts) + " attempts");
    auto end = hrClock::now();
    fDuration duration = end - start;
    info("Time elapsed: " + str(duration.count()) + " seconds.");
    return true;
}

bool DtccSolar::CalcVisProjMatrix(Rays *rays, fArray2D &visMatrix, fArray2D &projMatrix, fArray2D &visProjMatrix)
{
    int rayCount = rays->GetRayCount();

    for (int i = 0; i < mFaceCount; i++)
    {
        if (mFaceMask[i])
        {
            for (int j = 0; j < rayCount; j++)
            {
                // Calculate the projection matrix for each face
                visProjMatrix[i][j] = visMatrix[i][j] * projMatrix[i][j];
            }
        }
    }

    info("Visibility-Projection matrix calculated successfully.");
    return true;
}

bool DtccSolar::CalcIrradiance2Phase(Rays *rays, fArray2D &skySunMatrix, fArray2D &visProjMatrix, fArray2D &irradianceMatrix)
{
    auto vpShape = GetShape(visProjMatrix);
    auto ssMatShape = GetShape(skySunMatrix);

    info("Vis-Proj-Matrix has shape: (" + str(vpShape.first) + ", " + str(vpShape.second) + ")");
    info("Sky-Sun-Matrix has shape: (" + str(ssMatShape.first) + ", " + str(ssMatShape.second) + ")");
    info("Irradiance matrix shape: (" + str(vpShape.first) + ", " + str(ssMatShape.second) + ")");

    int timeSteps = ssMatShape.second;
    int rayCount = rays->GetRayCount();

    if (ssMatShape.first != rayCount)
    {
        error("Matrix shape mismatch. Array does not match rays. Cannot calculate irradiance.");
        return false;
    }

    if (vpShape.second != ssMatShape.first)
    {
        error("Matrix shape mismatch. Cannot calculate irradiance.");
        return false;
    }

    MatrixXfRM VP = VectorToEigen(visProjMatrix);
    MatrixXfRM S = VectorToEigen(skySunMatrix);
    MatrixXfRM E(mFaceCount, timeSteps);

    auto start = hrClock::now();
    E.noalias() = VP * S;
    auto end = hrClock::now();
    fDuration duration = end - start;

    irradianceMatrix = EigenToVector(E);
    info("Irradiance calculation with Eigen completed in " + str(duration.count()) + " seconds.");
    return true;
}

bool DtccSolar::CalcIrradiance3Phase(Rays *skyRays, Rays *sunRays, fArray2D &skyMatrix, fArray2D &sunMatrix, fArray2D &skyVisProjMatrix, fArray2D &sunVisProjMatrix, fArray2D &skyIrrMatrix, fArray2D &sunIrrMatrix)
{
    auto skyMatShape = GetShape(skyMatrix);
    auto sunMatShape = GetShape(sunMatrix);

    auto skyVisProjShape = GetShape(skyVisProjMatrix);
    auto sunVisProjShape = GetShape(sunVisProjMatrix);

    auto skyIrrShape = GetShape(skyIrrMatrix);
    auto sunIrrShape = GetShape(sunIrrMatrix);

    info("Sky-Matrix has shape: (" + str(skyMatShape.first) + ", " + str(skyMatShape.second) + ")");
    info("Sky-Vis-Proj-Matrix has shape: (" + str(skyVisProjShape.first) + ", " + str(skyVisProjShape.second) + ")");
    info("Sky-Irradiance-Matrix has shape: (" + str(skyIrrShape.first) + ", " + str(skyIrrShape.second) + ")");

    info("Sun-Matrix has shape: (" + str(sunMatShape.first) + ", " + str(sunMatShape.second) + ")");
    info("Sun-Vis-Proj-Matrix has shape: (" + str(sunVisProjShape.first) + ", " + str(sunVisProjShape.second) + ")");
    info("Sun-Irradiance-Matrix has shape: (" + str(sunIrrShape.first) + ", " + str(sunIrrShape.second) + ")");

    int skyTimeSteps = skyMatShape.second;

    if (skyMatShape.first != skyRays->GetRayCount() || sunMatShape.first != sunRays->GetRayCount())
    {
        error("Matrix shape mismatch. Cannot calculate irradiance.");
        return false;
    }

    MatrixXfRM skyVP = VectorToEigen(skyVisProjMatrix);
    MatrixXfRM skyS = VectorToEigen(skyMatrix);
    MatrixXfRM skyE(mFaceCount, skyTimeSteps);

    auto start1 = hrClock::now();
    skyE.noalias() = skyVP * skyS;
    auto end1 = hrClock::now();
    fDuration durationSky = end1 - start1;
    info("Irradiance from sky calculated with Eigen in " + str(durationSky.count()) + " seconds.");

    MatrixXfRM sunVP = VectorToEigen(sunVisProjMatrix);
    MatrixXfRM sunS = VectorToEigen(sunMatrix);
    MatrixXfRM sunE(mFaceCount, skyTimeSteps);

    // Since sunS is a diagonal matrix, we can optimize the multiplication
    VectorXf diagSun = sunS.diagonal();
    auto start2 = hrClock::now();
    sunE.noalias() = sunVP * diagSun.asDiagonal();
    auto end2 = hrClock::now();
    fDuration durationSun = end2 - start2;
    info("Irradiance from sun calculated with Eigen in " + str(durationSun.count()) + " seconds.");

    skyIrrMatrix = EigenToVector(skyE);
    sunIrrMatrix = EigenToVector(sunE);

    return true;
}

bool DtccSolar::Run2PhaseAnalysis(fArray2D sunSkyMat)
{
    info("-----------------------------------------------------");
    info("Running 2-phase analysis: E = VP * S");

    int numRays = mSunSkyRays->GetRayCount();

    fArray2D projMatrix = fArray2D(mFaceCount, fArray1D(numRays, 0.0f));
    fArray2D visMatrix = fArray2D(mFaceCount, fArray1D(numRays, 1.0f));
    fArray2D visProjMatrix = fArray2D(mFaceCount, fArray1D(numRays, 0.0f));
    fArray2D irrMatrix = fArray2D(mFaceCount, fArray1D(sunSkyMat[0].size(), 0.0f));

    // Calculate projection matrix
    if (!CalcProjMatrix(mSunSkyRays, projMatrix))
        return false;

    // Calculate visibility matrix
    if (!CalcVisMatrix(mSunSkyRays, visMatrix))
        return false;

    // Calculate the visibility-projection matrix
    if (!CalcVisProjMatrix(mSunSkyRays, visMatrix, projMatrix, visProjMatrix))
        return false;

    // Calculate irradiance
    if (!CalcIrradiance2Phase(mSunSkyRays, sunSkyMat, visProjMatrix, irrMatrix))
        return false;

    // Store the matrices
    mProjMatrixTot = projMatrix;
    mVisMatrixTot = visMatrix;
    mVisProjMatrixTot = visProjMatrix;
    mIrrMatrixTot = irrMatrix;

    info("2-phase analysis completed successfully.");
    info("-----------------------------------------------------");

    return true;
}

bool DtccSolar::Run3PhaseAnalysis(fArray2D skyMatrix, fArray2D sunMatrix)
{
    info("-----------------------------------------------------");
    info("Running 3-phase analysis: E = VP_sky * S_sky + VP_sun * S_sun");

    int numSkyRays = mSkyRays->GetRayCount();
    int numSunRays = mSunRays->GetRayCount();

    info("Number of sky rays: " + str(numSkyRays));
    info("Number of sun rays: " + str(numSunRays));

    fArray2D skyProjMatrix = fArray2D(mFaceCount, fArray1D(numSkyRays, 0.0f));
    fArray2D sunProjMatrix = fArray2D(mFaceCount, fArray1D(numSunRays, 0.0f));

    fArray2D skyVisMatrix = fArray2D(mFaceCount, fArray1D(numSkyRays, 1.0f));
    fArray2D sunVisMatrix = fArray2D(mFaceCount, fArray1D(numSunRays, 1.0f));

    fArray2D skyVisProjMatrix = fArray2D(mFaceCount, fArray1D(numSkyRays, 0.0f));
    fArray2D sunVisProjMatrix = fArray2D(mFaceCount, fArray1D(numSunRays, 0.0f));

    fArray2D skyIrrMatrix = fArray2D(mFaceCount, fArray1D(skyMatrix[0].size(), 0.0f));
    fArray2D sunIrrMatrix = fArray2D(mFaceCount, fArray1D(sunMatrix[0].size(), 0.0f));

    // Calculate sky projection matrix
    if (!CalcProjMatrix(mSkyRays, skyProjMatrix))
        return false;

    // Calculate sun projection matrix
    if (!CalcProjMatrix(mSunRays, sunProjMatrix))
        return false;

    // Calculate sky visibility matrix
    if (!CalcVisMatrix(mSkyRays, skyVisMatrix))
        return false;

    // Calculate sun visibility matrix
    if (!CalcVisMatrix(mSunRays, sunVisMatrix))
        return false;

    // Calculate the sky visibility-projection matrix
    if (!CalcVisProjMatrix(mSkyRays, skyVisMatrix, skyProjMatrix, skyVisProjMatrix))
        return false;

    // Calculate the sun visibility-projection matrix
    if (!CalcVisProjMatrix(mSunRays, sunVisMatrix, sunProjMatrix, sunVisProjMatrix))
        return false;

    // Calculate irradiance
    if (!CalcIrradiance3Phase(mSkyRays, mSunRays, skyMatrix, sunMatrix, skyVisProjMatrix, sunVisProjMatrix, skyIrrMatrix, sunIrrMatrix))
        return false;

    // Store the matrices for later retrieval
    mProjMatrixSky = skyProjMatrix;
    mProjMatrixSun = sunProjMatrix;
    mVisMatrixSky = skyVisMatrix;
    mVisMatrixSun = sunVisMatrix;
    mVisProjMatrixSky = skyVisProjMatrix;
    mVisProjMatrixSun = sunVisProjMatrix;

    mIrrMatrixSky = skyIrrMatrix;
    mIrrMatrixSun = sunIrrMatrix;

    info("3-phase analysis completed successfully.");
    info("-----------------------------------------------------");
    return true;
}

#ifdef PYTHON_MODULE

namespace py = pybind11;

PYBIND11_MODULE(py_solar, m)
{
    py::class_<DtccSolar>(m, "PySolar")
        .def(py::init<>())
        .def(py::init<std::vector<std::vector<float>>, std::vector<std::vector<int>>>())
        .def(py::init<std::vector<std::vector<float>>, std::vector<std::vector<int>>, std::vector<bool>, std::vector<std::vector<float>>, std::vector<float>>())
        .def(py::init<std::vector<std::vector<float>>, std::vector<std::vector<int>>, std::vector<bool>, std::vector<std::vector<float>>, std::vector<float>, std::vector<std::vector<float>>, std::vector<float>>())
        .def("get_mesh_faces", [](DtccSolar &self)
             { py::array out = py::cast(self.GetMeshFaces()); return out; })
        .def("get_mesh_vertices", [](DtccSolar &self)
             { py::array out = py::cast(self.GetMeshVertices()); return out; })
        .def("get_face_normals", [](DtccSolar &self)
             { py::array out = py::cast(self.GetFaceNormals()); return out; })
        .def("run_2_phase_analysis", [](DtccSolar &self, std::vector<std::vector<float>> sun_sky_mat)
             { py::array out = py::cast(self.Run2PhaseAnalysis(sun_sky_mat)); return out; })
        .def("run_3_phase_analysis", [](DtccSolar &self, std::vector<std::vector<float>> sky_mat, std::vector<std::vector<float>> sun_mat)
             { py::array out = py::cast(self.Run3PhaseAnalysis(sky_mat, sun_mat)); return out; })
        .def("get_visibility_matrix_tot", [](DtccSolar &self)
             { py::array out = py::cast(self.GetVisibilityMatrixTot()); return out; })
        .def("get_projection_matrix_tot", [](DtccSolar &self)
             { py::array out = py::cast(self.GetProjectionMatrixTot()); return out; })
        .def("get_irradiance_matrix_tot", [](DtccSolar &self)
             { py::array out = py::cast(self.GetIrradianceMatrixTot()); return out; })
        .def("get_visibility_vector_tot", [](DtccSolar &self)
             { py::array out = py::cast(self.GetVisibilityVectorTot()); return out; })
        .def("get_projection_vector_tot", [](DtccSolar &self)
             { py::array out = py::cast(self.GetProjectionVectorTot()); return out; })
        .def("get_irradiance_vector_tot", [](DtccSolar &self)
             { py::array out = py::cast(self.GetIrradianceVectorTot()); return out; })
        .def("get_visibility_matrix_sky", [](DtccSolar &self)
             { py::array out = py::cast(self.GetVisibilityMatrixSky()); return out; })
        .def("get_projection_matrix_sky", [](DtccSolar &self)
             { py::array out = py::cast(self.GetProjectionMatrixSky()); return out; })
        .def("get_irradiance_matrix_sky", [](DtccSolar &self)
             { py::array out = py::cast(self.GetIrradianceMatrixSky()); return out; })
        .def("get_visibility_vector_sky", [](DtccSolar &self)
             { py::array out = py::cast(self.GetVisibilityVectorSky()); return out; })
        .def("get_projection_vector_sky", [](DtccSolar &self)
             { py::array out = py::cast(self.GetProjectionVectorSky()); return out; })
        .def("get_irradiance_vector_sky", [](DtccSolar &self)
             { py::array out = py::cast(self.GetIrradianceVectorSky()); return out; })
        .def("get_visibility_matrix_sun", [](DtccSolar &self)
             { py::array out = py::cast(self.GetVisibilityMatrixSun()); return out; })
        .def("get_projection_matrix_sun", [](DtccSolar &self)
             { py::array out = py::cast(self.GetProjectionMatrixSun()); return out; })
        .def("get_irradiance_matrix_sun", [](DtccSolar &self)
             { py::array out = py::cast(self.GetIrradianceMatrixSun()); return out; })
        .def("get_visibility_vector_sun", [](DtccSolar &self)
             { py::array out = py::cast(self.GetVisibilityVectorSun()); return out; })
        .def("get_projection_vector_sun", [](DtccSolar &self)
             { py::array out = py::cast(self.GetProjectionVectorSun()); return out; })
        .def("get_irradiance_vector_sun", [](DtccSolar &self)
             { py::array out = py::cast(self.GetIrradianceVectorSun()); return out; });
}

#endif
