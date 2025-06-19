#include "embree_solar.h"

EmbreeSolar::EmbreeSolar()
{
    info("Creating embree instance with default constructor.");

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

    info("Model setup with plane geometry contais:");
    info("Number of vertices: " + str(mVertexCount));
    info("Number of faces: " + str(mFaceCount));

    mMaskCount = mFaceCount;
    mApplyMask = false;
    mFaceMask = std::vector<bool>(mFaceCount, true);
    mFaceNormals = new Vector[mFaceCount];

    CreateDevice();
    CreateScene();
    CreateGeomPlane();
    CalcFaceMidPoints();
    CalcFaceNormals();

    info("Model setup with plane geometry complete.");
}

EmbreeSolar::EmbreeSolar(fArray2D vertices, iArray2D faces)
{
    info("Creating embree instance with mesh geometry.");

    set_log_level(INFO);

    mVertexCount = (int)vertices.size();
    mFaceCount = (int)faces.size();
    mFaceNormals = new Vector[mFaceCount];

    mMaskCount = mFaceCount;
    mApplyMask = false;
    mFaceMask = std::vector<bool>(mFaceCount, true);

    CreateDevice();
    CreateScene();
    CreateGeom(vertices, faces);
    CalcFaceMidPoints();
    CalcFaceNormals();

    info("Model setup with mesh geometry complete.");
}

EmbreeSolar::EmbreeSolar(fArray2D vertices, iArray2D faces, bArray1D faceMask, fArray2D sunSkyRays, fArray1D solidAngles)
{
    info("Creating embree instance with mesh geometry.");
    set_log_level(INFO);

    Eigen::setNbThreads(std::thread::hardware_concurrency());
    Eigen::initParallel();
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

    CreateDevice();
    CreateScene();
    CreateGeom(vertices, faces);
    CalcFaceMidPoints();
    CalcFaceNormals();

    mSunSkyRays = new Rays(sunSkyRays, solidAngles);

    info("Model setup with mesh geometry complete.");
}

EmbreeSolar::EmbreeSolar(fArray2D vertices, iArray2D faces, bArray1D faceMask, fArray2D skyRays, fArray1D solidAngles, fArray2D sunRays)
{
    info("Creating embree instance with mesh geometry.");
    set_log_level(INFO);

    Eigen::setNbThreads(std::thread::hardware_concurrency());
    Eigen::initParallel();
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

    CreateDevice();
    CreateScene();
    CreateGeom(vertices, faces);
    CalcFaceMidPoints();
    CalcFaceNormals();

    mSkyRays = new Rays(skyRays, solidAngles);
    mSunRays = new Rays(sunRays);

    info("Model setup with mesh geometry complete.");
}

EmbreeSolar::~EmbreeSolar()
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
    if (mFaces != nullptr)
    {
        delete[] mFaces;
        mFaces = nullptr;
    }

    if (mVertices != nullptr)
    {
        delete[] mVertices;
        mVertices = nullptr;
    }

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

    // Release Embree geometry, scene, and device
    if (mGeometry)
        rtcReleaseGeometry(mGeometry);

    if (mScene)
        rtcReleaseScene(mScene);

    if (mDevice)
        rtcReleaseDevice(mDevice);
}

iArray2D EmbreeSolar::GetMeshFaces()
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

fArray2D EmbreeSolar::GetMeshVertices()
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

fArray2D EmbreeSolar::GetFaceNormals()
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

fArray2D EmbreeSolar::GetVisibilityResults()
{
    return mVisibilityMatrix;
}

fArray2D EmbreeSolar::GetProjectionResults()
{
    return mProjectionMatrix;
}

fArray2D EmbreeSolar::GetIrradianceResults()
{
    return mIrradianceMatrix;
}

void EmbreeSolar::CreateDevice()
{
    mDevice = rtcNewDevice(NULL);

    if (!mDevice)
        printf("error %d: cannot create device\n", rtcGetDeviceError(NULL));

    // rtcSetDeviceErrorFunction(mDevice, errorFunction, NULL);

    info("Device created.");
}

void EmbreeSolar::ErrorFunction(void *userPtr, enum RTCError error, const char *str)
{
    printf("error %d: %s\n", error, str);
}

void EmbreeSolar::CreateScene()
{
    mScene = rtcNewScene(mDevice);

    info("Scene created.");
}

void EmbreeSolar::CreateGeom(fArray2D vertices, iArray2D faces)
{
    mGeometry = rtcNewGeometry(mDevice, RTC_GEOMETRY_TYPE_TRIANGLE);

    mVertices = (Vertex *)rtcSetNewGeometryBuffer(mGeometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex), mVertexCount);
    mFaces = (Face *)rtcSetNewGeometryBuffer(mGeometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Face), mFaceCount);

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
            error("Invalid vertex size in EmbreeSolar::createGeom.");
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
            error("Invalid face size in EmbreeSolar::createGeom.");
    }

    rtcCommitGeometry(mGeometry);
    rtcAttachGeometry(mScene, mGeometry);
    rtcReleaseGeometry(mGeometry);
    rtcCommitScene(mScene);

    info("Geometry created from vertices and faces.");
}

void EmbreeSolar::CreateGeomPlane()
{
    /* create triangle mesh */
    const float xStep = (mPp.xMax - mPp.xMin) / (mPp.xCount - 1);
    const float yStep = (mPp.yMax - mPp.yMin) / (mPp.yCount - 1);

    const int nVertices = mPp.xCount * mPp.yCount;
    const int nFaces = (mPp.xCount - 1) * (mPp.yCount - 1) * 2;

    mGeometry = rtcNewGeometry(mDevice, RTC_GEOMETRY_TYPE_TRIANGLE);

    /* map triangle and vertex buffers */
    mVertices = (Vertex *)rtcSetNewGeometryBuffer(mGeometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex), nVertices);
    mFaces = (Face *)rtcSetNewGeometryBuffer(mGeometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Face), nFaces);

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

    rtcCommitGeometry(mGeometry);
    rtcAttachGeometry(mScene, mGeometry);
    rtcReleaseGeometry(mGeometry);
    rtcCommitScene(mScene);
}

void EmbreeSolar::CalcFaceMidPoints()
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

void EmbreeSolar::CalcFaceNormals()
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

bool EmbreeSolar::CalcProjMatrix(Rays *rays, fArray2D &mProjectionMatrix)
{
    if (!rays)
    {
        error("RayDome is not initialized. Cannot compute projection matrix.");
        return false;
    }

    fArray2D surfaceNormals = GetFaceNormals();
    fArray2D rayDirections = rays->GetRayDirections();
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
                mProjectionMatrix[i][j] = std::max(0.0f, dot);
            }
        }
    }

    info("Projection matrix was calculated successfully.");

    return true;
}

bool EmbreeSolar::CalcVisMatrix_Occ1(Rays *rays, fArray2D &visMatrix)
{
    int hitCounter = 0;
    int hitAttempts = 0;
    float hitPortion = 0.0f;
    float thisPortion = 0.0f;
    // Compute diffuse sky portion by iterating over all faces in the mesh
    // mVisibilityMatrix = fArray2D(mFaceCount, fArray1D(mPydome->GetRayCount(), 1.0f));
    mSkyViewFactor = fArray1D(mFaceCount, 0);
    auto start = hrClock::now();
    info("Calculating visibility matrix with rtcOccluded1 for " + str(mMaskCount) + " faces and " + str(rays->GetRayCount()) + " rays.");
    for (int i = 0; i < mFaceCount; i++)
    {
        if (mFaceMask[i])
        {
            rays->TranslateRays(mFaceMidPts[i]);
            int nRays = rays->GetRayCount();
            hitPortion = 0.0;
            for (int j = 0; j < nRays; j++)
            {
                RTCRay ray = rays->GetRays()[j];
                rtcOccluded1(mScene, &ray);

                if (ray.tfar == -std::numeric_limits<float>::infinity())
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
                info("Sky raytracing for " + str(i) + " faces completed.");
        }
    }

    info("Visibility matrix calculated successfully. Found " + str(hitCounter) + " intersections in " + str(hitAttempts) + " attempts.");
    auto end = hrClock::now();
    fDuration duration = end - start;
    info("Time elapsed: " + str(duration.count()) + " seconds.");
    return true;
}

bool EmbreeSolar::CalcVisMatrix_Occ8(Rays *rays, fArray2D &visMatrix)
{
    int hitCounter = 0;
    int hitAttempts = 0;
    float hitPortion = 0.0f;
    float thisPortion = 0.0f;
    float raySolidAngle = 0.0f;
    mSkyViewFactor = fArray1D(mFaceCount, 0);

    auto start = hrClock::now();
    info("Calculating visibility matrix with rtcOccluded8 for " + str(mMaskCount) + " faces and " + str(rays->GetRayCount()) + " rays.");
    for (int i = 0; i < mFaceCount; i++)
    {
        if (mFaceMask[i])
        {
            rays->Translate8Rays(mFaceMidPts[i]);
            int nBundles = rays->GetBundle8Count();
            hitPortion = 0.0;
            for (int j = 0; j < nBundles; j++)
            {
                RTCRay8 rayBundle = rays->GetRays8()[j];
                const int *valid = rays->GetValid8()[j];
                rtcOccluded8(valid, mScene, &rayBundle);
                for (int k = 0; k < 8; k++)
                {
                    int rayIndex = j * 8 + k;
                    if (rayBundle.tfar[k] == -std::numeric_limits<float>::infinity())
                    {
                        hitCounter++;
                        raySolidAngle = rays->GetSolidAngles()[rayIndex];
                        thisPortion = raySolidAngle / mDomeSolidAngle;
                        hitPortion = hitPortion + thisPortion;
                        visMatrix[i][rayIndex] = 0.0f;
                    }
                    hitAttempts++;
                }
            }
            mSkyViewFactor[i] = 1.0 - hitPortion;
            if (i > 0 && i % 10000 == 0)
                info("Sky raytracing for " + str(i) + " faces completed.");
        }
    }

    info("Visibility matrix calculated successfully. Found " + str(hitCounter) + " intersections in " + str(hitAttempts) + " attempts.");
    auto end = hrClock::now();
    fDuration duration = end - start;
    info("Time elapsed: " + str(duration.count()) + " seconds.");
    return true;
}

bool EmbreeSolar::CalcVisProjMatrix(Rays *rays, fArray2D &visMatrix, fArray2D &projMatrix, fArray2D &visProjMatrix)
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

bool EmbreeSolar::CalcIrradiance2Phase(Rays *rays, fArray2D &skySunMatrix, fArray2D &visProjMatrix, fArray2D &irradianceMatrix)
{
    auto ssMatShape = GetShape(skySunMatrix);
    info("Input array has shape row: " + str(ssMatShape.first) + " columns: " + str(ssMatShape.second));
    auto vpShape = GetShape(visProjMatrix);
    info("Vis-Proj matrix has shape row: " + str(vpShape.first) + " columns: " + str(vpShape.second));

    int timeSteps = ssMatShape.second;
    int rayCount = rays->GetRayCount();

    if (ssMatShape.first != rayCount)
    {
        error("Matrix shape missmatch. Array does not match rays. Cannot calculate irradiance.");
        return false;
    }

    if (vpShape.second != ssMatShape.first)
    {
        error("Matrix shape missmatch. Cannot calculate irradiance.");
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
    info("Irradiance calculation with Eigne completed in " + str(duration.count()) + " seconds.");
    return true;
}

bool EmbreeSolar::CalcIrradiance3Phase(Rays *skyRays, Rays *sunRays, fArray2D &skyMatrix, fArray2D &sunMatrix, fArray2D &skyVisProjMatrix, fArray2D &sunVisProjMatrix, fArray2D &irrMatrix)
{
    auto skyMatShape = GetShape(skyMatrix);
    auto sunMatShape = GetShape(sunMatrix);

    int skyTimeSteps = skyMatShape.second;
    // int sunTimeSteps = sunMatShape.second;

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
    info("Irradiance from sky calculated with Eigne in " + str(durationSky.count()) + " seconds.");

    MatrixXfRM sunVP = VectorToEigen(sunVisProjMatrix);
    MatrixXfRM sunS = VectorToEigen(sunMatrix);
    MatrixXfRM sunE(mFaceCount, skyTimeSteps);

    auto start2 = hrClock::now();
    skyE.noalias() = skyVP * skyS;
    auto end2 = hrClock::now();
    fDuration durationSun = end2 - start2;
    info("Irradiance from sky calculated with Eigne in " + str(durationSun.count()) + " seconds.");

    MatrixXfRM E(mFaceCount, skyTimeSteps);
    // Combine the two irradiance matrices
    E.noalias() = skyE + sunE;
    irrMatrix = EigenToVector(E);

    return true;
}

bool EmbreeSolar::Run2PhaseAnalysis(fArray2D sunSkyMat)
{
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
    if (!CalcVisMatrix_Occ1(mSunSkyRays, visMatrix))
        return false;

    // Calculate the visibility-projection matrix
    if (!CalcVisProjMatrix(mSunSkyRays, visMatrix, projMatrix, visProjMatrix))
        return false;

    // Calculate irradiance
    if (!CalcIrradiance2Phase(mSunSkyRays, sunSkyMat, visProjMatrix, irrMatrix))
        return false;

    info("2-phase analysis completed successfully.");

    // Store the matrices
    mProjectionMatrix = projMatrix;
    mVisibilityMatrix = visMatrix;
    mVisProjMatrix = visProjMatrix;
    mIrradianceMatrix = irrMatrix;

    return true;
}

bool EmbreeSolar::Run3PhaseAnalysis(fArray2D skyMatrix, fArray2D sunMatrix)
{
    info("Running 3-phase analysis: E = VP_sky * S_sky + VP_sun * S_sun");

    int numSkyRays = mSkyRays->GetRayCount();
    int numSunRays = mSunRays->GetRayCount();

    fArray2D skyProjMatrix = fArray2D(mFaceCount, fArray1D(numSkyRays, 0.0f));
    fArray2D sunProjMatrix = fArray2D(mFaceCount, fArray1D(numSunRays, 0.0f));

    fArray2D skyVisMatrix = fArray2D(mFaceCount, fArray1D(numSkyRays, 1.0f));
    fArray2D sunVisMatrix = fArray2D(mFaceCount, fArray1D(numSunRays, 1.0f));

    fArray2D skyVisProjMatrix = fArray2D(mFaceCount, fArray1D(numSkyRays, 0.0f));
    fArray2D sunVisProjMatrix = fArray2D(mFaceCount, fArray1D(numSunRays, 0.0f));

    fArray2D irrMatrix = fArray2D(mFaceCount, fArray1D(skyMatrix[0].size(), 0.0f));

    // Calculate sky projection matrix
    if (!CalcProjMatrix(mSkyRays, skyProjMatrix))
        return false;

    // Calculate sun projection matrix
    if (!CalcProjMatrix(mSunRays, sunProjMatrix))
        return false;

    // Calculate sky visibility matrix
    if (!CalcVisMatrix_Occ1(mSkyRays, skyVisMatrix))
        return false;

    // Calculate sun visibility matrix
    if (!CalcVisMatrix_Occ1(mSunRays, sunVisMatrix))
        return false;

    // Calculate the sky visibility-projection matrix
    if (!CalcVisProjMatrix(mSkyRays, skyVisMatrix, skyProjMatrix, skyVisProjMatrix))
        return false;

    // Calculate the sun visibility-projection matrix
    if (!CalcVisProjMatrix(mSunRays, sunVisMatrix, sunProjMatrix, sunVisProjMatrix))
        return false;

    // Calculate irradiance
    if (!CalcIrradiance3Phase(mSkyRays, mSunRays, skyMatrix, sunMatrix, skyVisProjMatrix, sunVisProjMatrix, irrMatrix))
        return false;

    info("2-phase analysis completed successfully.");
    return true;
}

#ifdef PYTHON_MODULE

namespace py = pybind11;

PYBIND11_MODULE(py_embree_solar, m)
{
    py::class_<EmbreeSolar>(m, "PyEmbreeSolar")
        .def(py::init<>())
        .def(py::init<std::vector<std::vector<float>>, std::vector<std::vector<int>>>())
        .def(py::init<std::vector<std::vector<float>>, std::vector<std::vector<int>>, std::vector<bool>, std::vector<std::vector<float>>, std::vector<float>>())
        .def(py::init<std::vector<std::vector<float>>, std::vector<std::vector<int>>, std::vector<bool>, std::vector<std::vector<float>>, std::vector<float>, std::vector<std::vector<float>>>())
        .def("get_mesh_faces", [](EmbreeSolar &self)
             { py::array out = py::cast(self.GetMeshFaces()); return out; })
        .def("get_mesh_vertices", [](EmbreeSolar &self)
             { py::array out = py::cast(self.GetMeshVertices()); return out; })
        .def("get_face_normals", [](EmbreeSolar &self)
             { py::array out = py::cast(self.GetFaceNormals()); return out; })
        .def("run_2_phase_analysis", [](EmbreeSolar &self, std::vector<std::vector<float>> sun_sky_mat)
             { py::array out = py::cast(self.Run2PhaseAnalysis(sun_sky_mat)); return out; })
        .def("run_3_phase_analysis", [](EmbreeSolar &self, std::vector<std::vector<float>> sky_mat, std::vector<std::vector<float>> sun_mat)
             { py::array out = py::cast(self.Run3PhaseAnalysis(sky_mat, sun_mat)); return out; })
        .def("get_irradiance_results", [](EmbreeSolar &self)
             { py::array out = py::cast(self.GetIrradianceResults()); return out; })
        .def("get_visibility_results", [](EmbreeSolar &self)
             { py::array out = py::cast(self.GetVisibilityResults()); return out; })
        .def("get_projection_results", [](EmbreeSolar &self)
             { py::array out = py::cast(self.GetProjectionResults()); return out; });
}

#endif