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

    mSkydome = new Skydome();
    mSunrays = new Sunrays();

    info("Model setup with plane geometry complete.");
}

EmbreeSolar::EmbreeSolar(std::vector<std::vector<float>> vertices, std::vector<std::vector<int>> faces)
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

    mSkydome = new Skydome(10);
    mSunrays = new Sunrays(mFaceMidPts, mFaceCount, mFaceMask);

    info("Model setup with mesh geometry complete.");
}

EmbreeSolar::EmbreeSolar(std::vector<std::vector<float>> vertices, std::vector<std::vector<int>> faces, std::vector<bool> faceMask)
{
    info("Creating embree instance with mesh geometry.");
    set_log_level(INFO);

    mVertexCount = (int)vertices.size();
    mFaceCount = (int)faces.size();
    mFaceNormals = new Vector[mFaceCount];

    mApplyMask = true;
    mFaceMask = faceMask;
    mMaskCount = 0;
    for (int i = 0; i < mFaceCount; i++)
        if (mFaceMask[i])
            mMaskCount++;

    CreateDevice();
    CreateScene();
    CreateGeom(vertices, faces);
    CalcFaceMidPoints();
    CalcFaceNormals();

    mSkydome = new Skydome(10);
    mSunrays = new Sunrays(mFaceMidPts, mFaceCount, mFaceMask);

    mHasSunResults = false;
    mHasSkyResults = false;
    mHasIrrResults = false;

    info("Model setup with mesh geometry complete.");
}

EmbreeSolar::EmbreeSolar(std::vector<std::vector<float>> vertices, std::vector<std::vector<int>> faces, std::vector<bool> faceMask, int skyType)
{
    info("Creating embree instance with mesh geometry.");
    set_log_level(INFO);

    mVertexCount = (int)vertices.size();
    mFaceCount = (int)faces.size();
    mFaceNormals = new Vector[mFaceCount];

    mApplyMask = false;
    mMaskCount = 0;
    mFaceMask = std::vector<bool>(mFaceCount, true);
    mMaskCount = 0;
    for (int i = 0; i < mFaceCount; i++)
        if (mFaceMask[i])
            mMaskCount++;

    CreateDevice();
    CreateScene();
    CreateGeom(vertices, faces);
    CalcFaceMidPoints();
    CalcFaceNormals();

    mSkydome = new Skydome(skyType);
    mSunrays = new Sunrays(mFaceMidPts, mFaceCount, mFaceMask);

    mHasSunResults = false;
    mHasSkyResults = false;
    mHasIrrResults = false;

    info("Model setup with mesh geometry complete.");
}

EmbreeSolar::~EmbreeSolar()
{
    delete mSkydome;
    delete mSunrays;

    delete[] mFaceMidPts;
    delete[] mFaces;
    delete[] mVertices;
    delete[] mFaceNormals;

    rtcReleaseScene(mScene);
    rtcReleaseDevice(mDevice);

    printf("Destructor called.\n");
}

std::vector<std::vector<int>> EmbreeSolar::GetMeshFaces()
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

std::vector<std::vector<float>> EmbreeSolar::GetMeshVertices()
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

std::vector<std::vector<float>> EmbreeSolar::GetFaceNormals()
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

std::vector<std::vector<int>> EmbreeSolar::GetSkydomeFaces()
{
    return mSkydome->GetFaces();
}

std::vector<std::vector<float>> EmbreeSolar::GetSkydomeVertices()
{
    return mSkydome->GetVertices();
}

std::vector<std::vector<float>> EmbreeSolar::GetSkydomeRayDirections()
{
    return mSkydome->GetRayDirections();
}

std::vector<std::vector<int>> EmbreeSolar::GetOccludedResults()
{
    return mOccluded;
}

std::vector<std::vector<float>> EmbreeSolar::GetAngleResults()
{
    return mAngles;
}

std::vector<std::vector<int>> EmbreeSolar::GetFaceSkyHitResults()
{
    return mFaceSkyHit;
}

std::vector<float> EmbreeSolar::GetSkyViewFactorResults()
{
    return mSkyViewFactor;
}

std::vector<float> EmbreeSolar::GetIrradianceResultsDNI()
{
    return mIrradianceDNI;
}

std::vector<float> EmbreeSolar::GetIrradianceResultsDHI()
{
    return mIrradianceDHI;
}

std::vector<float> EmbreeSolar::GetAccumulatedAngles()
{
    return mAccumAngles;
}

std::vector<float> EmbreeSolar::GetAccumulatedOcclusion()
{
    return mAccumOcclud;
}

int EmbreeSolar::GetSkydomeRayCount()
{
    return mSkydome->GetRayCount();
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

void EmbreeSolar::CreateGeom(std::vector<std::vector<float>> vertices, std::vector<std::vector<int>> faces)
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
        mFaceNormals[i] = vNormal;
    }
}

void EmbreeSolar::Raytrace_occ1(std::vector<float> &angles, std::vector<int> &occluded, int &hitCounter)
{
    int nRays = mSunrays->GetRayCount();
    for (int i = 0; i < nRays; i++)
    {
        if (mFaceMask[i])
        {
            RTCRay ray = mSunrays->GetRays()[i];
            rtcOccluded1(mScene, &ray);

            if (ray.tfar == -std::numeric_limits<float>::infinity())
            {
                occluded[i] = 1;
                hitCounter++;
            }

            // RayIndex is the same as FaceIndex
            Vector vec = {ray.dir_x, ray.dir_y, ray.dir_z};
            Vector nml = mFaceNormals[i];
            float angle = CalcAngle2(vec, nml);
            angles[i] = angle;
        }
    }
}

void EmbreeSolar::Raytrace_occ4(std::vector<float> &angles, std::vector<int> &occluded, int &hitCounter)
{
    int nBundles = mSunrays->GetBundle4Count();
    for (int i = 0; i < nBundles; i++)
    {
        RTCRay4 rayBundle = mSunrays->GetRays4()[i];
        const int *valid = mSunrays->GetValid4(mApplyMask)[i];
        rtcOccluded4(valid, mScene, &rayBundle);
        for (int j = 0; j < 4; j++)
        {
            int rayIndex = i * 4 + j;
            if (rayBundle.tfar[j] == -std::numeric_limits<float>::infinity())
            {
                hitCounter++;
                occluded[rayIndex] = 1;
            }

            if (valid[j] == -1) // If ray is valid
            {
                // RayIndex is the same as FaceIndex
                Vector ray = {rayBundle.dir_x[j], rayBundle.dir_y[j], rayBundle.dir_z[j]};
                Vector nml = mFaceNormals[rayIndex];
                float angle = CalcAngle2(ray, nml);
                angles[rayIndex] = angle;
            }
        }
    }
}

void EmbreeSolar::Raytrace_occ8(std::vector<float> &angles, std::vector<int> &occluded, int &hitCounter)
{
    int nBundles = mSunrays->GetBundle8Count();
    for (int i = 0; i < nBundles; i++)
    {
        RTCRay8 rayBundle = mSunrays->GetRays8()[i];
        const int *valid = mSunrays->GetValid8(mApplyMask)[i];
        rtcOccluded8(valid, mScene, &rayBundle);

        for (int j = 0; j < 8; j++)
        {
            int rayIndex = i * 8 + j;
            if (rayBundle.tfar[j] == -std::numeric_limits<float>::infinity())
            {
                occluded[rayIndex] = 1;
                hitCounter++;
            }

            if (valid[j] == -1) // If ray is valid
            {
                // RayIndex is the same as FaceIndex
                Vector ray = {rayBundle.dir_x[j], rayBundle.dir_y[j], rayBundle.dir_z[j]};
                Vector nml = mFaceNormals[rayIndex];
                float angle = CalcAngle2(ray, nml);
                angles[rayIndex] = angle;
            }
        }
    }
}

void EmbreeSolar::Raytrace_occ16(std::vector<float> &angles, std::vector<int> &occluded, int &hitCounter)
{
    int nBundles = mSunrays->GetBundle16Count();
    for (int i = 0; i < nBundles; i++)
    {
        RTCRay16 rayBundle = mSunrays->GetRays16()[i];
        const int *valid = mSunrays->GetValid16(mApplyMask)[i];
        rtcOccluded16(valid, mScene, &rayBundle);

        for (int j = 0; j < 16; j++)
        {
            int rayIndex = i * 16 + j;
            if (rayBundle.tfar[j] == -std::numeric_limits<float>::infinity())
            {
                hitCounter++;
                occluded[rayIndex] = 1;
            }

            if (valid[j] == -1) // If ray is valid
            {
                // RayIndex is the same as FaceIndex
                Vector ray = {rayBundle.dir_x[j], rayBundle.dir_y[j], rayBundle.dir_z[j]};
                Vector nml = mFaceNormals[rayIndex];
                float angle = CalcAngle2(ray, nml);
                angles[rayIndex] = angle;
            }
        }
    }
}

bool EmbreeSolar::SunRaytrace_Occ1(std::vector<std::vector<float>> sun_vecs)
{
    // Define a 2D vector to store intersection results. Each postion is given the
    // initial values 0, which is changed to 1 if an intersection is found.
    int nSunVecs = (int)sun_vecs.size();
    mAngles = std::vector<std::vector<float>>(nSunVecs, std::vector<float>(mFaceCount, 0.0f));
    mOccluded = std::vector<std::vector<int>>(nSunVecs, std::vector<int>(mFaceCount, 0));

    auto start = std::chrono::high_resolution_clock::now();
    int hitCounter = 0;

    info("Running rtcOccluded1 for " + str(nSunVecs) + " sun vectors.");
    for (int i = 0; i < nSunVecs; i++)
    {
        auto sun_vec = UnitizeVector(sun_vecs[i]);
        auto &angels = mAngles[i];
        auto &occluded = mOccluded[i];

        if (sun_vec.size() == 3)
        {
            mSunrays->UpdateRay1Directions(sun_vec, mApplyMask);
            Raytrace_occ1(angels, occluded, hitCounter);
        }
        else
        {
            error("Invalid sun vector size in EmbreeSolar::iterateRaytrace_occ1.");
            return false;
        }
        if (i > 0 && i % 100 == 0)
            info("Sun raytracing for " + str(i) + " sun vectors completed.");
    }

    info("Found " + str(hitCounter) + " intersections.");
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    info("Time elapsed: " + str(duration.count()) + " seconds.");
    mHasSunResults = true;
    return true;
}

bool EmbreeSolar::SunRaytrace_Occ4(std::vector<std::vector<float>> sun_vecs)
{
    // Define a 2D vector to store intersection results. Each postion is given the
    // initial values 0, which is changed to 1 if an intersection is found.
    int nSunVecs = (int)sun_vecs.size();
    mAngles = std::vector<std::vector<float>>(nSunVecs, std::vector<float>(mFaceCount, 0.0));
    mOccluded = std::vector<std::vector<int>>(nSunVecs, std::vector<int>(mFaceCount, 0));
    auto start = std::chrono::high_resolution_clock::now();
    int hitCounter = 0;

    info("Running rtcOccluded4 for " + str(nSunVecs) + " sun vectors.");
    for (int i = 0; i < nSunVecs; i++)
    {
        auto sun_vec = UnitizeVector(sun_vecs[i]);
        auto &angels = mAngles[i];
        auto &occluded = mOccluded[i];
        if (sun_vec.size() == 3)
        {
            mSunrays->UpdateRay4Directions(sun_vec, mApplyMask);
            Raytrace_occ4(angels, occluded, hitCounter);
        }
        else
        {
            error("Invalid sun vector size in EmbreeSolar::iterateRaytrace_occ4.");
            return false;
        }
        if (i > 0 && i % 100 == 0)
            info("Sun raytracing for " + str(i) + " sun vectors completed.");
    }

    info("Found " + str(hitCounter) + " intersections.");
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    info("Time elapsed: " + str(duration.count()) + " seconds.");
    mHasSunResults = true;
    return true;
}

bool EmbreeSolar::SunRaytrace_Occ8(std::vector<std::vector<float>> sun_vecs)
{
    // Define a 2D vector to store intersection results. Each postion is given the
    // initial values 0, which is changed to 1 if an intersection is found.
    int nSunVecs = (int)sun_vecs.size();
    mAngles = std::vector<std::vector<float>>(nSunVecs, std::vector<float>(mFaceCount, 0.0));
    mOccluded = std::vector<std::vector<int>>(nSunVecs, std::vector<int>(mFaceCount, 0));

    auto start = std::chrono::high_resolution_clock::now();
    int hitCounter = 0;

    info("Running rtcOccluded8 for " + str(nSunVecs) + " sun vectors.");
    for (int i = 0; i < nSunVecs; i++)
    {
        auto sun_vec = UnitizeVector(sun_vecs[i]);
        auto &angels = mAngles[i];
        auto &occluded = mOccluded[i];

        if (sun_vec.size() == 3)
        {
            mSunrays->UpdateRay8Directions(sun_vec, mApplyMask);
            Raytrace_occ8(angels, occluded, hitCounter);
        }
        else
        {
            error("Invalid sun vector size in EmbreeSolar::iterateRaytrace_occ8.");
            return false;
        }
        if (i > 0 && i % 100 == 0)
            info("Sun raytracing for " + str(i) + " sun vectors completed.");
    }

    info("Found " + str(hitCounter) + " intersections.");
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    info("Time elapsed: " + str(duration.count()) + " seconds.");
    mHasSunResults = true;
    return true;
}

bool EmbreeSolar::SunRaytrace_Occ16(std::vector<std::vector<float>> sun_vecs)
{
    // Define a 2D vector to store intersection results. Each postion is given the
    // initial values 0, which is changed to 1 if an intersection is found.
    int nSunVecs = (int)sun_vecs.size();
    mAngles = std::vector<std::vector<float>>(nSunVecs, std::vector<float>(mFaceCount, 0.0));
    mOccluded = std::vector<std::vector<int>>(nSunVecs, std::vector<int>(mFaceCount, 0));

    auto start = std::chrono::high_resolution_clock::now();
    info("Running rtcOccluded16 for " + str(nSunVecs) + " sun vectors.");
    int hitCounter = 0;
    for (int i = 0; i < nSunVecs; i++)
    {
        auto sun_vec = UnitizeVector(sun_vecs[i]);
        auto &angles = mAngles[i];
        auto &occluded = mOccluded[i];
        if (sun_vec.size() == 3)
        {
            mSunrays->UpdateRay16Directions(sun_vec, mApplyMask);
            Raytrace_occ16(angles, occluded, hitCounter);
        }
        else
        {
            error("Invalid sun vector size in EmbreeSolar::iterateRaytrace_occ16.");
            return false;
        }
        if (i > 0 && i % 100 == 0)
            info("Sun raytracing for " + str(i) + " sun vectors completed.");
    }

    info("Found " + str(hitCounter) + " intersections.");
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    info("Time elapsed: " + str(duration.count()) + " seconds.");
    mHasSunResults = true;
    return true;
}

bool EmbreeSolar::SkyRaytrace_Occ1()
{
    int hitCounter = 0;
    int hitAttempts = 0;
    float hitPortion = 0.0f;
    // Compute diffuse sky portion by iterating over all faces in the mesh
    mFaceSkyHit = std::vector<std::vector<int>>(mFaceCount, std::vector<int>(mSkydome->GetRayCount(), 0));
    mSkyViewFactor = std::vector<float>(mFaceCount, 0);
    auto start = std::chrono::high_resolution_clock::now();
    info("Running diffuse rtcOccluded1 for " + str(mMaskCount) + " faces and " + str(mSkydome->GetRayCount()) + " skydome rays.");
    for (int i = 0; i < mFaceCount; i++)
    {
        if (mFaceMask[i])
        {
            mSkydome->TranslateRays(mFaceMidPts[i]);
            int nRays = mSkydome->GetRayCount();
            hitPortion = 0.0;
            for (int j = 0; j < nRays; j++)
            {
                RTCRay ray = mSkydome->GetRays()[j];
                rtcOccluded1(mScene, &ray);
                if (ray.tfar == -std::numeric_limits<float>::infinity())
                {
                    hitCounter++;
                    hitPortion = hitPortion + mSkydome->GetRayAreas()[j];
                    mFaceSkyHit[i][j] = 1;
                }
                hitAttempts++;
            }
            mSkyViewFactor[i] = 1.0 - hitPortion;
            if (i > 0 && i % 10000 == 0)
                info("Sky raytracing for " + str(i) + " faces completed.");
        }
    }

    info("Found " + str(hitCounter) + " intersections in " + str(hitAttempts) + " attempts.");
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    info("Time elapsed: " + str(duration.count()) + " seconds.");
    mHasSkyResults = true;
    return true;
}

bool EmbreeSolar::SkyRaytrace_Occ4()
{
    int hitCounter = 0;
    int hitAttempts = 0;
    float hitPortion = 0.0f;
    // Compute diffuse sky portion by iterating over all faces in the mesh
    mFaceSkyHit = std::vector<std::vector<int>>(mFaceCount, std::vector<int>(mSkydome->GetRayCount(), 0));
    mSkyViewFactor = std::vector<float>(mFaceCount, 0);
    auto start = std::chrono::high_resolution_clock::now();
    info("Running diffuse rtcOccluded4 for " + str(mMaskCount) + " faces and " + str(mSkydome->GetRayCount()) + " skydome rays.");
    for (int i = 0; i < mFaceCount; i++)
    {
        if (mFaceMask[i])
        {
            mSkydome->Translate4Rays(mFaceMidPts[i]);
            int nBundles = mSkydome->GetBundle4Count();
            hitPortion = 0.0;
            for (int j = 0; j < nBundles; j++)
            {
                RTCRay4 rayBundle = mSkydome->GetRays4()[j];
                const int *valid = mSkydome->GetValid4()[j];
                rtcOccluded4(valid, mScene, &rayBundle);
                for (int k = 0; k < 4; k++)
                {
                    int rayIndex = j * 4 + k;
                    if (rayBundle.tfar[k] == -std::numeric_limits<float>::infinity())
                    {
                        hitCounter++;
                        hitPortion = hitPortion + mSkydome->GetRayAreas()[rayIndex];
                        mFaceSkyHit[i][rayIndex];
                    }
                    hitAttempts++;
                }
            }
            mSkyViewFactor[i] = 1.0 - hitPortion;
            if (i > 0 && i % 10000 == 0)
                info("Sky raytracing for " + str(i) + " faces completed.");
        }
    }

    info("Found " + str(hitCounter) + " intersections in " + str(hitAttempts) + " attempts.");
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    info("Time elapsed: " + str(duration.count()) + " seconds.");
    mHasSkyResults = true;
    return true;
}

bool EmbreeSolar::SkyRaytrace_Occ8()
{
    int hitCounter = 0;
    int hitAttempts = 0;
    float hitPortion = 0.0f;
    // Compute diffuse sky portion by iterating over all faces in the mesh
    mFaceSkyHit = std::vector<std::vector<int>>(mFaceCount, std::vector<int>(mSkydome->GetRayCount(), 0));
    mSkyViewFactor = std::vector<float>(mFaceCount, 0);
    auto start = std::chrono::high_resolution_clock::now();
    info("Running diffuse rtcOccluded8 for " + str(mMaskCount) + " faces and " + str(mSkydome->GetRayCount()) + " skydome rays.");
    for (int i = 0; i < mFaceCount; i++)
    {
        if (mFaceMask[i])
        {
            mSkydome->Translate8Rays(mFaceMidPts[i]);
            int nBundles = mSkydome->GetBundle8Count();
            hitPortion = 0.0;
            for (int j = 0; j < nBundles; j++)
            {
                RTCRay8 rayBundle = mSkydome->GetRays8()[j];
                const int *valid = mSkydome->GetValid8()[j];
                rtcOccluded8(valid, mScene, &rayBundle);
                for (int k = 0; k < 8; k++)
                {
                    int rayIndex = j * 8 + k;
                    if (rayBundle.tfar[k] == -std::numeric_limits<float>::infinity())
                    {
                        hitCounter++;
                        hitPortion = hitPortion + mSkydome->GetRayAreas()[rayIndex];
                        mFaceSkyHit[i][rayIndex] = 1;
                    }
                    hitAttempts++;
                }
            }
            mSkyViewFactor[i] = 1.0 - hitPortion;
            if (i > 0 && i % 10000 == 0)
                info("Sky raytracing for " + str(i) + " faces completed.");
        }
    }

    info("Found " + str(hitCounter) + " intersections in " + str(hitAttempts) + " attempts.");
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    info("Time elapsed: " + str(duration.count()) + " seconds.");
    mHasSkyResults = true;
    return true;
}

bool EmbreeSolar::SkyRaytrace_Occ16()
{
    int hitCounter = 0;
    int hitAttempts = 0;
    float hitPortion = 0.0f;
    // Compute diffuse sky portion by iterating over all faces in the mesh
    mFaceSkyHit = std::vector<std::vector<int>>(mFaceCount, std::vector<int>(mSkydome->GetRayCount(), 0));
    mSkyViewFactor = std::vector<float>(mFaceCount, 0);
    auto start = std::chrono::high_resolution_clock::now();
    info("Running diffuse rtcOccluded16 for " + str(mMaskCount) + " faces and " + str(mSkydome->GetRayCount()) + " skydome rays.");
    for (int i = 0; i < mFaceCount; i++)
    {
        if (mFaceMask[i])
        {
            mSkydome->Translate16Rays(mFaceMidPts[i]);
            int nBundles = mSkydome->GetBundle16Count();
            hitPortion = 0.0;
            for (int j = 0; j < nBundles; j++)
            {
                RTCRay16 rayBundle = mSkydome->GetRays16()[j];
                const int *valid = mSkydome->GetValid16()[j];
                rtcOccluded16(valid, mScene, &rayBundle);
                for (int k = 0; k < 16; k++)
                {
                    int rayIndex = j * 16 + k;
                    if (rayBundle.tfar[k] == -std::numeric_limits<float>::infinity())
                    {
                        hitCounter++;
                        hitPortion = hitPortion + mSkydome->GetRayAreas()[rayIndex];
                        mFaceSkyHit[i][rayIndex] = 1;
                    }
                    hitAttempts++;
                }
            }
            mSkyViewFactor[i] = 1.0 - hitPortion;
            if (i > 0 && i % 10000 == 0)
                info("Sky raytracing for " + str(i) + " faces completed.");
        }
    }

    info("Found " + str(hitCounter) + " intersections in " + str(hitAttempts) + " attempts.");
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    info("Time elapsed: " + str(duration.count()) + " seconds.");
    mHasSkyResults = true;
    return true;
}

bool EmbreeSolar::CalcIrradiance(std::vector<float> dni, std::vector<float> dhi)
{
    mIrradianceDNI = std::vector<float>(mFaceCount, 0.0);
    mIrradianceDHI = std::vector<float>(mFaceCount, 0.0);
    mAccumAngles = std::vector<float>(mFaceCount, 0.0);
    mAccumOcclud = std::vector<float>(mFaceCount, 0.0);

    int nDNI = (int)dni.size();
    int nDHI = (int)dhi.size();

    if (nDNI < 1 || nDHI < 1)
    {
        error("Invalid weather data count in EmbreeSolar::CalcIrradiance.");
        return false;
    }
    else if (mHasSunResults && (nDNI != (int)mAngles.size()))
    {
        error("Weather data does not match number of suns used for raytracing in EmbreeSolar::CalcIrradiance.");
        return false;
    }
    else // Data size is correct
    {
        for (long unsigned int sunIndex = 0; sunIndex < mAngles.size(); sunIndex++)
        {
            float weatherDNI = dni[sunIndex]; // Direct normal irradiance
            float weatherDHI = dhi[sunIndex]; // Direct horizontal irradiance

            for (long unsigned int faceIndex = 0; faceIndex < mAngles[sunIndex].size(); faceIndex++)
            {
                if (mHasSunResults)
                {
                    float angle = M_PI - mAngles[sunIndex][faceIndex];
                    float angleFraction = angle / M_PI;
                    float faceInSun = 1.0f - mOccluded[sunIndex][faceIndex];
                    float irradianceDNI = weatherDNI * faceInSun * angleFraction;
                    mIrradianceDNI[faceIndex] += irradianceDNI;
                    mAccumAngles[faceIndex] += mAngles[sunIndex][faceIndex];
                    mAccumOcclud[faceIndex] += mOccluded[sunIndex][faceIndex];
                }
                if (mHasSkyResults)
                {
                    float svf = mSkyViewFactor[faceIndex];
                    float irradianceDHI = weatherDHI * svf;
                    mIrradianceDHI[faceIndex] += irradianceDHI;
                }
            }
        }
        mHasIrrResults = true;
    }
    info("Irradiance calculation completed.");
    return true;
}

bool EmbreeSolar::CalcIrradianceGroup(std::vector<float> dni, std::vector<float> dhi, std::vector<std::vector<int>> sunGroups)
{
    mIrradianceDNI = std::vector<float>(mFaceCount, 0.0);
    mIrradianceDHI = std::vector<float>(mFaceCount, 0.0);
    mAccumAngles = std::vector<float>(mFaceCount, 0.0);
    mAccumOcclud = std::vector<float>(mFaceCount, 0.0);

    for (long unsigned int i = 0; i < sunGroups.size(); i++)
    {
        int groupSunIndex = i;

        for (long unsigned int j = 0; j < sunGroups[i].size(); j++)
        {
            // Weather data is stored for the real sun positions
            int realSunIndex = sunGroups[i][j];
            float weatherDNI = dni[realSunIndex]; // Direct normal irradiance
            float weatherDHI = dhi[realSunIndex]; // Direct horizontal irradiance

            // Angles and occluded are calculated for the group sun positions
            for (long unsigned int faceIndex = 0; faceIndex < mAngles[groupSunIndex].size(); faceIndex++)
            {
                if (mHasSunResults)
                {
                    float angle = M_PI - mAngles[groupSunIndex][faceIndex];
                    float angleFraction = angle / M_PI;
                    float faceInSun = 1.0f - mOccluded[groupSunIndex][faceIndex];
                    float irradianceDNI = weatherDNI * faceInSun * angleFraction;
                    mIrradianceDNI[faceIndex] += irradianceDNI;
                    mAccumAngles[faceIndex] += mAngles[groupSunIndex][faceIndex];
                    mAccumOcclud[faceIndex] += mOccluded[groupSunIndex][faceIndex];
                }

                if (mHasSkyResults)
                {
                    float svf = mSkyViewFactor[faceIndex];
                    float irradianceDHI = weatherDHI * svf;
                    mIrradianceDHI[faceIndex] += irradianceDHI;
                }
            }
        }
    }
    mHasIrrResults = true;
    info("Irradiance calculation completed.");
    return true;
}

#ifdef PYTHON_MODULE

namespace py = pybind11;

PYBIND11_MODULE(py_embree_solar, m)
{
    py::class_<EmbreeSolar>(m, "PyEmbreeSolar")
        .def(py::init<>())
        .def(py::init<std::vector<std::vector<float>>, std::vector<std::vector<int>>>())
        .def(py::init<std::vector<std::vector<float>>, std::vector<std::vector<int>>, std::vector<bool>>())
        .def(py::init<std::vector<std::vector<float>>, std::vector<std::vector<int>>, std::vector<bool>, int>())
        .def("get_mesh_faces", [](EmbreeSolar &self)
             { py::array out = py::cast(self.GetMeshFaces()); return out; })
        .def("get_mesh_vertices", [](EmbreeSolar &self)
             { py::array out = py::cast(self.GetMeshVertices()); return out; })
        .def("get_face_normals", [](EmbreeSolar &self)
             { py::array out = py::cast(self.GetFaceNormals()); return out; })
        .def("get_skydome_faces", [](EmbreeSolar &self)
             { py::array out = py::cast(self.GetSkydomeFaces()); return out; })
        .def("get_skydome_vertices", [](EmbreeSolar &self)
             { py::array out = py::cast(self.GetSkydomeVertices()); return out; })
        .def("get_skydome_rays", [](EmbreeSolar &self)
             { py::array out = py::cast(self.GetSkydomeRayDirections()); return out; })
        .def("get_skydome_ray_count", [](EmbreeSolar &self)
             { py::array out = py::cast(self.GetSkydomeRayCount()); return out; })
        .def("sun_raytrace_occ1", [](EmbreeSolar &self, std::vector<std::vector<float>> sun_vecs)
             { py::array out = py::cast(self.SunRaytrace_Occ1(sun_vecs)); return out; })
        .def("sun_raytrace_occ4", [](EmbreeSolar &self, std::vector<std::vector<float>> sun_vecs)
             { py::array out = py::cast(self.SunRaytrace_Occ4(sun_vecs)); return out; })
        .def("sun_raytrace_occ8", [](EmbreeSolar &self, std::vector<std::vector<float>> sun_vecs)
             { py::array out = py::cast(self.SunRaytrace_Occ8(sun_vecs)); return out; })
        .def("sun_raytrace_occ16", [](EmbreeSolar &self, std::vector<std::vector<float>> sun_vecs)
             { py::array out = py::cast(self.SunRaytrace_Occ16(sun_vecs)); return out; })
        .def("sky_raytrace_occ1", [](EmbreeSolar &self)
             { py::array out = py::cast(self.SkyRaytrace_Occ1()); return out; })
        .def("sky_raytrace_occ4", [](EmbreeSolar &self)
             { py::array out = py::cast(self.SkyRaytrace_Occ4()); return out; })
        .def("sky_raytrace_occ8", [](EmbreeSolar &self)
             { py::array out = py::cast(self.SkyRaytrace_Occ8()); return out; })
        .def("sky_raytrace_occ16", [](EmbreeSolar &self)
             { py::array out = py::cast(self.SkyRaytrace_Occ16()); return out; })
        .def("calc_irradiance", [](EmbreeSolar &self, std::vector<float> dni, std::vector<float> dhi)
             { py::array out = py::cast(self.CalcIrradiance(dni, dhi)); return out; })
        .def("calc_irradiance_group", [](EmbreeSolar &self, std::vector<float> dni, std::vector<float> dhi, std::vector<std::vector<int>> sunGroups)
             { py::array out = py::cast(self.CalcIrradianceGroup(dni, dhi, sunGroups)); return out; })
        .def("get_angle_results", [](EmbreeSolar &self)
             { py::array out = py::cast(self.GetAngleResults()); return out; })
        .def("get_occluded_results", [](EmbreeSolar &self)
             { py::array out = py::cast(self.GetOccludedResults()); return out; })
        .def("get_face_skyhit_results", [](EmbreeSolar &self)
             { py::array out = py::cast(self.GetFaceSkyHitResults()); return out; })
        .def("get_sky_view_factor_results", [](EmbreeSolar &self)
             { py::array out = py::cast(self.GetSkyViewFactorResults()); return out; })
        .def("get_results_dni", [](EmbreeSolar &self)
             { py::array out = py::cast(self.GetIrradianceResultsDNI()); return out; })
        .def("get_results_dhi", [](EmbreeSolar &self)
             { py::array out = py::cast(self.GetIrradianceResultsDHI()); return out; })
        .def("get_accumulated_angles", [](EmbreeSolar &self)
             { py::array out = py::cast(self.GetAccumulatedAngles()); return out; })
        .def("get_accumulated_occlusion", [](EmbreeSolar &self)
             { py::array out = py::cast(self.GetAccumulatedOcclusion()); return out; });
}

#endif