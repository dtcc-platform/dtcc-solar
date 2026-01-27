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

MatrixXfRM DtccSolar::GetVPMatrix()
{
    return mVPMatrix;
}

MatrixXfRM &DtccSolar::GetIrradianceMatrix()
{
    return mIrrMatrix;
}

VectorXf DtccSolar::GetIrradianceMatrixFlat()
{
    return RowSums(mIrrMatrix);
}

VectorXf DtccSolar::GetIrradianceVector()
{
    return mIrrVector;
}

// Sky results

MatrixXfRM DtccSolar::GetVPMatrixSky()
{
    return mVPMatrixSky;
}

MatrixXfRM &DtccSolar::GetIrradianceMatrixSky()
{
    return mIrrMatrixSky;
}

VectorXf DtccSolar::GetIrradianceMatrixSkyFlat()
{
    return RowSums(mIrrMatrixSky);
}

MatrixXfRM DtccSolar::GetVPMatrixSun()
{
    return mVPMatrixSun;
}

MatrixXfRM &DtccSolar::GetIrradianceMatrixSun()
{
    return mIrrMatrixSun;
}

VectorXf DtccSolar::GetIrradianceMatrixSunFlat()
{
    return RowSums(mIrrMatrixSun);
}

VectorXf DtccSolar::GetIrradianceVectorSun()
{
    return mIrrVectorSun;
}

VectorXf DtccSolar::GetIrradianceVectorSky()
{
    return mIrrVectorSky;
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

bool DtccSolar::CalcProjMatrix(Rays *rays, fArray2D &mProjectionMatrix, fArray2D &surfaceNormals)
{
    if (!rays)
    {
        error("RayDome is not initialized. Cannot compute projection matrix.");
        return false;
    }

    fArray2D rayDirections = rays->GetRayDirections();
    fArray1D raySolidAngles = rays->GetSolidAngles();
    size_t numRays = rayDirections.size();
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

bool DtccSolar::CalcVisMatrix(Rays *rays, fArray2D &visMatrix, fArray2D &surfaceNormals)
{
    int hitCounter = 0;
    int hitAttempts = 0;
    float thisPortion = 0.0f;
    fArray2D rayDirs = rays->GetRayDirections();

    mSkyViewFactor = fArray1D(mFaceCount, 0);
    auto start = hrClock::now();
    info("Calculating visibility matrix with BVH for " + str(mMaskCount) + " faces and " + str(rays->GetRayCount()) + " rays.");

    static constexpr size_t stack_size = 64;
#pragma omp parallel for schedule(dynamic) reduction(+ : hitCounter, hitAttempts)
    for (int i = 0; i < mFaceCount; i++)
    {
        auto n = surfaceNormals[i];
        if (mFaceMask[i])
        {
            // rays->TranslateRays(mFaceMidPts[i]);
            Vec3 face_origin(mFaceMidPts[i].x, mFaceMidPts[i].y, mFaceMidPts[i].z);
            int nRays = rays->GetRayCount();
            float hitPortion = 0.0;

            for (int j = 0; j < nRays; j++)
            {
                const auto &r = rayDirs[j];

                // Dot product n · r (Lambert hemisphere test)
                const float dot = n[0] * r[0] + n[1] * r[1] + n[2] * r[2];

                // If ray is behind the face, it cannot contribute -> skip BVH
                if (dot <= 0.0f)
                {
                    visMatrix[i][j] = 0.0f;
                    continue;
                }

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
                            if (auto hit = mAccel->precomputed_tris[k].intersect(ray))
                            {
                                occluded = true;
                                return true;
                            }
                        }
                        return false;
                    });

                if (occluded)
                {
                    hitCounter++;
                    thisPortion = rays->GetSolidAngles()[j] / mDomeSolidAngle;
                    hitPortion += thisPortion;
                    visMatrix[i][j] = 0.0f; // override default 1.0f
                }

                hitAttempts++;
            }
            mSkyViewFactor[i] = 1.0 - hitPortion;
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
#pragma omp parallel for schedule(static)
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

    // Optional consistency check: VP must match (faces x rays)
    if (VP.rows() != mFaceCount || VP.cols() != rayCount)
    {
        error("VP dimensions do not match (mFaceCount x rayCount).");
        return false;
    }

    // Convert input to Eigen vector
    const Eigen::VectorXf S = VectorToEigen(skySunVector);

    // Ensure output has correct size
    if (E.size() != mFaceCount)
        E.resize(mFaceCount);

    auto start = hrClock::now();
    E.noalias() = VP * S;
    auto end = hrClock::now();

    info("Irradiance vector min: " + std::to_string(E.minCoeff()) + ", max: " + std::to_string(E.maxCoeff()));

    fDuration duration = end - start;
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

    // Logging: cast Eigen::Index to avoid str() overload ambiguity
    info("Vis-Proj-Matrix has shape: (" + str(static_cast<size_t>(faceCount)) + ", " + str(static_cast<size_t>(rayCountVP)) + ")");
    info("Sky-Sun-Matrix has shape: (" + str(static_cast<size_t>(rayCountSS)) + ", " + str(static_cast<size_t>(timeSteps)) + ")");
    info("Irradiance matrix shape: (" + str(static_cast<size_t>(faceCount)) + ", " + str(static_cast<size_t>(timeSteps)) + ")");

    const int rayCountRays = rays->GetRayCount();

    // Shape checks
    if (rayCountSS != rayCountRays)
    {
        error("Matrix shape mismatch. skySun rows do not match ray count. Cannot calculate irradiance.");
        return false;
    }

    if (rayCountVP != rayCountSS)
    {
        error("Matrix shape mismatch. visProj cols do not match skySun rows. Cannot calculate irradiance.");
        return false;
    }

    if (faceCount != mFaceCount)
    {
        error("Matrix shape mismatch. visProj rows do not match mFaceCount. Cannot calculate irradiance.");
        return false;
    }

    // Allocate output
    irradiance.resize(mFaceCount, static_cast<int>(timeSteps));

    // Compute E = VP * S
    auto start = hrClock::now();
    irradiance.noalias() = visProj * skySun;
    auto end = hrClock::now();
    fDuration duration = end - start;

    info("Irradiance calculation with Eigen completed in " + str(duration.count()) + " seconds.");
    return true;
}

bool DtccSolar::CalcIrradiance3Phase(Rays *skyRays, Rays *sunRays, VectorXf &skyS, VectorXf &sunS, const MatrixXfRM &skyVP, const MatrixXfRM &sunVP, VectorXf &Esky, VectorXf &Esun)
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

    // Shape checks
    if (skyVP.rows() != mFaceCount || sunVP.rows() != mFaceCount)
    {
        error("VP matrix row count does not match mFaceCount. Cannot calculate irradiance.");
        return false;
    }

    if (skyVP.cols() != skyRayCount)
    {
        error("Sky VP matrix column count does not match sky ray count.");
        return false;
    }

    if (sunVP.cols() != sunRayCount)
    {
        error("Sun VP matrix column count does not match sun ray count.");
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

    Esky.resize(mFaceCount);
    Esun.resize(mFaceCount);

    auto start = hrClock::now();

    Esky.noalias() = skyVP * skyS;
    Esun.noalias() = sunVP * sunS;

    auto end = hrClock::now();
    fDuration duration = end - start;

    info("3-phase irradiance vector calculation completed in " + str(duration.count()) + " seconds.");

    return true;
}

bool DtccSolar::CalcIrradiance3Phase(Rays *skyRays, Rays *sunRays, MatrixXfRM &skyS, MatrixXfRM &sunS, MatrixXfRM &skyVP, MatrixXfRM &sunVP, MatrixXfRM &skyE, MatrixXfRM &sunE)
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

    int skyTimeSteps = skyMatShape.second;

    if (skyMatShape.first != skyRays->GetRayCount() || sunMatShape.first != sunRays->GetRayCount())
    {
        error("Matrix shape mismatch. Cannot calculate irradiance.");
        return false;
    }

    skyE.resize(mFaceCount, skyTimeSteps);

    auto start1 = hrClock::now();
    skyE.noalias() = skyVP * skyS;
    info("skyE calculation done");
    auto end1 = hrClock::now();
    fDuration durationSky = end1 - start1;
    info("Irradiance from sky calculated with Eigen in " + str(durationSky.count()) + " seconds.");

    sunE.resize(mFaceCount, skyTimeSteps);

    // Since sunS is a diagonal matrix, we can optimize the multiplication
    VectorXf diagSun = sunS.diagonal();
    auto start2 = hrClock::now();
    sunE.noalias() = sunVP * diagSun.asDiagonal();
    auto end2 = hrClock::now();
    fDuration durationSun = end2 - start2;
    info("Irradiance from sun calculated with Eigen in " + str(durationSun.count()) + " seconds.");

    return true;
}

bool DtccSolar::CalcVPMatrix(Rays *rays, MatrixXfRM &visProj, fArray2D &surfaceNormals)
{
    if (!rays)
    {
        error("Rays is not initialized. Cannot compute visibility-projection matrix.");
        return false;
    }

    // Cache ray data
    const fArray2D rayDirs = rays->GetRayDirections();   // (nRays x 3)
    const fArray1D solidAngles = rays->GetSolidAngles(); // (nRays)
    const int nRays = rays->GetRayCount();

    // Allocate / resize contiguous VP matrix once
    if (visProj.rows() != mFaceCount || visProj.cols() != nRays)
        visProj.resize(mFaceCount, nRays);

    // Important: set to zero so masked faces and skipped rays are 0 without touching them
    visProj.setZero();

    // Sky view factor per face
    mSkyViewFactor = fArray1D(mFaceCount, 0.0f);

    int hitCounter = 0;
    int hitAttempts = 0;

    auto start = hrClock::now();
    info("Calculating visibility-projection matrix with BVH for " + str(mMaskCount) +
         " faces and " + str(nRays) + " rays.");

    static constexpr size_t stack_size = 64;

#pragma omp parallel for schedule(dynamic) reduction(+ : hitCounter, hitAttempts)
    for (int i = 0; i < mFaceCount; ++i)
    {
        if (!mFaceMask[i])
            continue;

        const auto &n = surfaceNormals[i];
        Vec3 face_origin(mFaceMidPts[i].x, mFaceMidPts[i].y, mFaceMidPts[i].z);

        float hitPortion = 0.0f;

        // Fast pointer to the start of the row (RowMajor: contiguous row storage)
        float *row = visProj.data() + static_cast<size_t>(i) * static_cast<size_t>(nRays);

        for (int j = 0; j < nRays; ++j)
        {
            const auto &r = rayDirs[j];

            // Lambert hemisphere test (and projection base)
            const float dot = n[0] * r[0] + n[1] * r[1] + n[2] * r[2];

            // Back-facing rays contribute nothing -> skip BVH, keep 0
            if (dot <= 0.0f)
                continue;

            const float proj = dot * solidAngles[j];

            // Build ray at face origin
            Ray ray = rays->GetRays()[j]; // copy
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
                hitPortion += (solidAngles[j] / mDomeSolidAngle);
                // row[j] stays 0
            }
            else
            {
                row[j] = proj; // visible => vis*proj = proj
            }
        }

        mSkyViewFactor[i] = 1.0f - hitPortion;
    }

    auto end = hrClock::now();
    fDuration duration = end - start;

    info("Visibility-projection matrix calculated successfully.");
    info("Found " + str(hitCounter) + " intersections in " + str(hitAttempts) + " attempts");
    info("Time elapsed: " + str(duration.count()) + " seconds.");

    return true;
}

bool DtccSolar::Run2PhaseAnalysis(fArray1D sunSkyVec)
{
    info("-----------------------------------------------------");
    info("Running 2-phase 1D analysis: E = VP * S");

    if (!mSunSkyRays)
    {
        error("mSunSkyRays is not initialized.");
        return false;
    }

    // Output vector (per face)
    VectorXf E;

    // Face normals (still fArray2D in your codebase)
    fArray2D surfaceNormals = GetFaceNormals();

    // Contiguous VP matrix (Eigen RowMajor) - avoids vector<vector<float>> penalty
    MatrixXfRM VP; // size will be set in CalcVPMatrix

    // Calculate VP matrix directly into contiguous storage
    if (!CalcVPMatrix(mSunSkyRays, VP, surfaceNormals))
        return false;

    // Calculate irradiance: E = VP * S (S is 1D vector)
    if (!CalcIrradiance2Phase(mSunSkyRays, sunSkyVec, VP, E))
        return false;

    // Store results
    // Recommended: store VP as Eigen to avoid converting back.
    // If you still need the old fArray2D for JSON/export/debug, convert only when needed.
    mVPMatrix = VP; // <-- add MatrixXfRM mVPMatrix; as a member
    mIrrVector = E;

    info("2-phase analysis completed successfully.");
    info("-----------------------------------------------------");

    return true;
}

bool DtccSolar::Run2PhaseAnalysis(fArray2D sunSkyMat)
{
    info("-----------------------------------------------------");
    info("Running 2-phase 2D analysis: E = VP * S");

    if (!mSunSkyRays)
    {
        error("mSunSkyRays is not initialized.");
        return false;
    }

    const int numRays = mSunSkyRays->GetRayCount();

    // Basic input validation (avoid sunSkyMat[0] crash)
    if (sunSkyMat.empty() || sunSkyMat[0].empty())
    {
        error("sunSkyMat is empty. Cannot run analysis.");
        return false;
    }

    // Check that sky-sun matrix row count matches rays (shape: rays x timesteps)
    if (static_cast<int>(sunSkyMat.size()) != numRays)
    {
        error("sunSkyMat row count does not match ray count.");
        return false;
    }

    // Face normals (still fArray2D in your codebase)
    fArray2D surfaceNormals = GetFaceNormals();

    // Contiguous VP matrix (Eigen RowMajor)
    MatrixXfRM VP; // sized inside CalcVPMatrix

    // Calculate VP matrix directly into contiguous storage
    if (!CalcVPMatrix(mSunSkyRays, VP, surfaceNormals))
        return false;

    // Convert sky-sun matrix to Eigen once (avoid repeated conversions elsewhere)
    auto start = hrClock::now();
    MatrixXfRM skySun = VectorToEigen(sunSkyMat); // (numRays x timeSteps)
    auto end = hrClock::now();
    fDuration duration = end - start;
    info("Converted sunSkyMat to Eigen in " + str(duration.count()) + " seconds.");

    // Output irradiance as Eigen matrix
    MatrixXfRM E; // (mFaceCount x timeSteps), resized inside CalcIrradiance2Phase

    // Calculate irradiance: E = VP * skySun
    if (!CalcIrradiance2Phase(mSunSkyRays, skySun, VP, E))
        return false;

    // Store results (recommended: keep Eigen types to avoid later conversions)
    mVPMatrix = VP; // add MatrixXfRM mVPMatrix; as a member
    mIrrMatrix = E; // add MatrixXfRM mIrrMatrix; as a member

    info("2-phase analysis completed successfully.");
    info("-----------------------------------------------------");
    return true;
}

bool DtccSolar::Run3PhaseAnalysis(fArray1D skyVector, fArray1D sunVector)
{
    info("-----------------------------------------------------");
    info("Running 3-phase 1D analysis: E = VP_sky * S_sky + VP_sun * S_sun");

    int numSkyRays = mSkyRays->GetRayCount();
    int numSunRays = mSunRays->GetRayCount();

    info("Number of sky rays: " + str(numSkyRays));
    info("Number of sun rays: " + str(numSunRays));

    // Contiguous VP matrix (Eigen RowMajor) - avoids vector<vector<float>> penalty
    MatrixXfRM skyVP; // size will be set in CalcVPMatrix
    MatrixXfRM sunVP; // size will be set in CalcVPMatrix

    VectorXf skyIrrVector;
    VectorXf sunIrrVector;

    fArray2D surfaceNormals = GetFaceNormals();

    VectorXf skyS = VectorToEigen(skyVector);
    VectorXf sunS = VectorToEigen(sunVector);

    // Calculate sky projection matrix
    if (!CalcVPMatrix(mSkyRays, skyVP, surfaceNormals))
        return false;

    // Calculate sun projection matrix
    if (!CalcVPMatrix(mSunRays, sunVP, surfaceNormals))
        return false;

    // Calculate irradiance
    if (!CalcIrradiance3Phase(mSkyRays, mSunRays, skyS, sunS, skyVP, sunVP, skyIrrVector, sunIrrVector))
        return false;

    // Store the matrices for later retrieval
    mVPMatrixSky = skyVP;
    mVPMatrixSun = sunVP;

    mIrrVectorSky = skyIrrVector;
    mIrrVectorSun = sunIrrVector;

    info("3-phase analysis completed successfully.");
    info("-----------------------------------------------------");
    return true;
}

bool DtccSolar::Run3PhaseAnalysis(fArray2D skyMatrix, fArray2D sunMatrix)
{
    info("-----------------------------------------------------");
    info("Running 3-phase 2D analysis: E = VP_sky * S_sky + VP_sun * S_sun");

    if (!mSkyRays || !mSunRays)
    {
        error("mSkyRays or mSunRays is not initialized.");
        return false;
    }

    const int numSkyRays = mSkyRays->GetRayCount();
    const int numSunRays = mSunRays->GetRayCount();

    info("Number of sky rays: " + str(numSkyRays));
    info("Number of sun rays: " + str(numSunRays));

    // Basic input validation (avoid skyMatrix[0] crashes)
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

    // Shape checks for inputs: (rayCount x timeSteps)
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

    // Face normals (still fArray2D in your codebase)
    fArray2D surfaceNormals = GetFaceNormals();

    // Convert sky/sun contribution matrices to Eigen once
    MatrixXfRM skyS = VectorToEigen(skyMatrix); // (numSkyRays x timeSteps)
    MatrixXfRM sunS = VectorToEigen(sunMatrix); // (numSunRays x timeSteps)

    // Compute VP matrices directly in contiguous Eigen storage
    MatrixXfRM skyVP;
    MatrixXfRM sunVP;

    if (!CalcVPMatrix(mSkyRays, skyVP, surfaceNormals))
        return false;

    if (!CalcVPMatrix(mSunRays, sunVP, surfaceNormals))
        return false;

    // Output irradiance matrices (Eigen)
    MatrixXfRM skyE;
    MatrixXfRM sunE;

    // Compute irradiance (two multiplications)
    if (!CalcIrradiance3Phase(mSkyRays, mSunRays, skyS, sunS, skyVP, sunVP, skyE, sunE))
        return false;

    // Store results (recommended: keep Eigen matrices for performance)
    mVPMatrixSky = skyVP; // MatrixXfRM member
    mVPMatrixSun = sunVP; // MatrixXfRM member
    mIrrMatrixSky = skyE;
    mIrrMatrixSun = sunE;

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

    return py::array(
        py::dtype::of<Scalar>(),
        {rows, cols},
        {stride_row, stride_col},
        M.data(),
        owner // keeps owning C++ object alive
    );
}

// Helper: copy Eigen::VectorXf -> numpy 1D (explicit stride, lifetime-safe)
inline py::array vec_to_numpy_1d(Eigen::VectorXf v)
{
    const py::ssize_t n = static_cast<py::ssize_t>(v.size());
    py::array_t<float> out(n);
    std::copy(v.data(), v.data() + v.size(), out.mutable_data());
    py::buffer_info info(out.mutable_data(), sizeof(float), py::format_descriptor<float>::format(), 1, {n}, {static_cast<py::ssize_t>(sizeof(float))});
    return py::array(info, out); // out is the base owner
}

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
        .def("run_2_phase_analysis_vec", [](DtccSolar &self, std::vector<float> sun_sky_vec)
             { py::array out = py::cast(self.Run2PhaseAnalysis(sun_sky_vec)); return out; })
        .def("run_2_phase_analysis_mat", [](DtccSolar &self, std::vector<std::vector<float>> sun_sky_mat)
             { py::array out = py::cast(self.Run2PhaseAnalysis(sun_sky_mat)); return out; })
        .def("run_3_phase_analysis_vec", [](DtccSolar &self, std::vector<float> sky_vec, std::vector<float> sun_vec)
             { py::array out = py::cast(self.Run3PhaseAnalysis(sky_vec, sun_vec)); return out; })
        .def("run_3_phase_analysis_mat", [](DtccSolar &self, std::vector<std::vector<float>> sky_mat, std::vector<std::vector<float>> sun_mat)
             { py::array out = py::cast(self.Run3PhaseAnalysis(sky_mat, sun_mat)); return out; })
        .def("get_irradiance_vector", [](DtccSolar &self)
             { return vec_to_numpy_1d(self.GetIrradianceVector()); })
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
             { return vec_to_numpy_1d(self.GetIrradianceMatrixSunFlat()); });
}

#endif
