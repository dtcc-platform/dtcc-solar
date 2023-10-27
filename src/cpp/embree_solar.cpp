#include "embree_solar.h"

EmbreeSolar::EmbreeSolar()
{
    // Plane parameters
    mPp.xMin = -10.0f;
    mPp.xMax = 10.0f;
    mPp.yMin = -10.0f;
    mPp.yMax = 10.0f;
    mPp.xPadding = 0.0f;
    mPp.yPadding = 0.0f;
    mPp.xCount = 201;
    mPp.yCount = 201;

    // Ray parameters
    mRp.xMin = -10.0f;
    mRp.xMax = 10.0f;
    mRp.yMin = -10.0f;
    mRp.yMax = 10.0f;
    mRp.xPadding = 0.1f;
    mRp.yPadding = 0.1f;
    mRp.xCount = 280;
    mRp.yCount = 280;

    mVertexCount = mPp.xCount * mPp.yCount;
    mFaceCount = (mPp.xCount - 1) * (mPp.yCount - 1) * 2;

    std::cout << "Model setup with plane geometry contais:" << std::endl;
    std::cout << "Number of vertices: " << mVertexCount << std::endl;
    std::cout << "Number of faces: " << mFaceCount << std::endl;

    int rayCount = mRp.xCount * mRp.yCount;

    initArrays(rayCount);
    createDevice();
    createScene();
    createGeomPlane();
    createGridRays();
    bundleRays();

    std::cout << "Model setup with predefinde settings complete." << std::endl;
}

EmbreeSolar::EmbreeSolar(std::vector<std::vector<float>> vertices, std::vector<std::vector<int>> faces, std::vector<float> sun_vec)
{
    mVertexCount = vertices.size();
    mFaceCount = faces.size();

    int rayCount = mFaceCount;
    initArrays(rayCount);
    createDevice();
    createScene();
    createGeom(vertices, faces);
    createRaysFromFaces(sun_vec);
    bundleRays();

    std::cout << "Model setup with mesh geometry complete." << std::endl;
}

EmbreeSolar::~EmbreeSolar()
{
    delete[] mRays;
    delete[] mRays4;
    delete[] mRays8;
    delete[] mRays16;

    delete[] mVertices;
    delete[] mFaces;
    delete[] mFaceMidPts;

    // Delete the 2d arrays
    for (int i = 0; i < mBundle4Count; i++)
        delete[] mRays4Valid[i];
    delete[] mRays4Valid;

    for (int i = 0; i < mBundle8Count; i++)
        delete[] mRays8Valid[i];
    delete[] mRays8Valid;

    for (int i = 0; i < mBundle16Count; i++)
        delete[] mRays16Valid[i];
    delete[] mRays16Valid;

    delete[] mVertices;
    delete[] mFaces;

    release();
}

void EmbreeSolar::release()
{
    rtcReleaseScene(mScene);
    rtcReleaseDevice(mDevice);
}

void EmbreeSolar::initArrays(int rayCount)
{
    mRayCount = rayCount;

    mBundle4Count = ceil((float)rayCount / 4.0f);
    mBundle8Count = ceil((float)rayCount / 8.0f);
    mBundle16Count = ceil((float)rayCount / 16.0f);

    std::cout << "Number of rays: " << rayCount << std::endl;
    std::cout << "Number of 4 bundles: " << mBundle4Count << std::endl;
    std::cout << "Number of 8 bundles: " << mBundle8Count << std::endl;
    std::cout << "Number of 16 bundles: " << mBundle16Count << std::endl;

    // Defining the arrays for the rays on the heap
    mRays = new RTCRay[rayCount];
    mRays4 = new RTCRay4[mBundle4Count];
    mRays8 = new RTCRay8[mBundle8Count];
    mRays16 = new RTCRay16[mBundle16Count];

    // Defining a 2d array for the vadility of each ray in the 4 group bundles.
    mRays4Valid = new int *[mBundle4Count];
    for (int i = 0; i < mBundle4Count; i++)
    {
        mRays4Valid[i] = new int[4];
        for (int j = 0; j < 4; j++)
            mRays4Valid[i][j] = 0;
    }

    // Defining a 2d array for the vadility of each ray in the 8 group bundles.
    mRays8Valid = new int *[mBundle8Count];
    for (int i = 0; i < mBundle8Count; i++)
    {
        mRays8Valid[i] = new int[8];
        for (int j = 0; j < 8; j++)
            mRays8Valid[i][j] = 0;
    }

    // Defining a 2d array for the vadility of each ray in the 16 group bundles.
    mRays16Valid = new int *[mBundle16Count];
    for (int i = 0; i < mBundle16Count; i++)
    {
        mRays16Valid[i] = new int[16];
        for (int j = 0; j < 16; j++)
            mRays16Valid[i][j] = 0;
    }
}

void EmbreeSolar::createDevice()
{
    mDevice = rtcNewDevice(NULL);

    if (!mDevice)
        printf("error %d: cannot create device\n", rtcGetDeviceError(NULL));

    // rtcSetDeviceErrorFunction(mDevice, errorFunction, NULL);

    printf("Device created.\n");
}

void EmbreeSolar::errorFunction(void *userPtr, enum RTCError error, const char *str)
{
    printf("error %d: %s\n", error, str);
}

void EmbreeSolar::createScene()
{
    mScene = rtcNewScene(mDevice);

    printf("Scene created.\n");
}

void EmbreeSolar::createGeom(std::vector<std::vector<float>> vertices, std::vector<std::vector<int>> faces)
{
    mGeometry = rtcNewGeometry(mDevice, RTC_GEOMETRY_TYPE_TRIANGLE);

    mVertices = (Vertex *)rtcSetNewGeometryBuffer(mGeometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof(Vertex), mVertexCount);
    mFaces = (Face *)rtcSetNewGeometryBuffer(mGeometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof(Face), mFaceCount);

    for (int i = 0; i < vertices.size(); i++)
    {
        if (vertices[i].size() == 3)
        {
            Vertex &v = mVertices[i];
            v.x = vertices[i][0];
            v.y = vertices[i][1];
            v.z = vertices[i][2];
        }
        else
            printf("Invalid vertex size in EmbreeSolar::createGeom.\n");
    }

    for (int i = 0; i < faces.size(); i++)
    {
        if (faces[i].size() == 3)
        {
            Face &f = mFaces[i];
            f.v0 = faces[i][0];
            f.v1 = faces[i][1];
            f.v2 = faces[i][2];
        }
        else
            printf("Invalid face size in EmbreeSolar::createGeom.\n");
    }

    rtcCommitGeometry(mGeometry);
    unsigned int geomID = rtcAttachGeometry(mScene, mGeometry);
    rtcReleaseGeometry(mGeometry);
    rtcCommitScene(mScene);

    printf("Geometry created from vertices and faces.\n");
}

void EmbreeSolar::createGeomPlane()
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
    unsigned int geomID = rtcAttachGeometry(mScene, mGeometry);
    rtcReleaseGeometry(mGeometry);
    rtcCommitScene(mScene);
}

void EmbreeSolar::createRaysFromFaces(std::vector<float> sun_vec)
{
    mFaceMidPts = new Vertex[mFaceCount];
    mRayCount = mFaceCount;

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

    // Create rays from face mid pts and sun vector

    for (int i = 0; i < mFaceCount; i++)
    {
        mRays[i].org_x = mFaceMidPts[i].x;
        mRays[i].org_y = mFaceMidPts[i].y;
        mRays[i].org_z = mFaceMidPts[i].z;

        mRays[i].dir_x = sun_vec[0];
        mRays[i].dir_y = sun_vec[1];
        mRays[i].dir_z = sun_vec[2];

        mRays[i].tnear = 0.05; // 5 cm
        mRays[i].tfar = std::numeric_limits<float>::infinity();
        mRays[i].mask = -1;
        mRays[i].flags = 0;
    }
}

void EmbreeSolar::createGridRays()
{
    float xStep = ((mRp.xMax - mRp.xPadding) - (mRp.xMin + mRp.xPadding)) / (mRp.xCount - 1);
    float yStep = ((mRp.yMax - mRp.yPadding) - (mRp.yMin + mRp.yPadding)) / (mRp.yCount - 1);

    int rayCounter = 0;

    /* create grid of rays within the bounds of the mesh */
    for (int i = 0; i < mRp.yCount; i++)
    {
        float y = (mRp.yMin + mRp.yPadding) + i * yStep;
        for (int j = 0; j < mRp.xCount; j++)
        {
            float x = (mRp.xMin + mRp.xPadding) + j * xStep;
            float z = -1.0f;

            mRays[rayCounter].org_x = x;
            mRays[rayCounter].org_y = y;
            mRays[rayCounter].org_z = z;

            mRays[rayCounter].dir_x = 0.0f;
            mRays[rayCounter].dir_y = 0.0f;
            mRays[rayCounter].dir_z = 1.0f;

            mRays[rayCounter].tnear = 0;
            mRays[rayCounter].tfar = std::numeric_limits<float>::infinity();
            mRays[rayCounter].mask = -1;
            mRays[rayCounter].flags = 0;

            rayCounter++;
        }
    }
}

void EmbreeSolar::bundleRays()
{
    int bundleIndex4 = -1;
    int rayIndex4 = 0;
    int bundleIndex8 = -1;
    int rayIndex8 = 0;
    int bundleIndex16 = -1;
    int rayIndex16 = 0;

    /* Sort the rays in groups of 4, 8 and 16 */
    for (int i = 0; i < mRayCount; i++)
    {
        float x = mRays[i].org_x;
        float y = mRays[i].org_y;
        float z = mRays[i].org_z;

        float dir_x = mRays[i].dir_x;
        float dir_y = mRays[i].dir_y;
        float dir_z = mRays[i].dir_z;

        float tNear = mRays[i].tnear;
        float tFar = mRays[i].tfar;
        unsigned int mask = mRays[i].mask;
        unsigned int flag = mRays[i].flags;

        rayIndex4 = i % 4;

        if (rayIndex4 == 0)
            bundleIndex4++;

        // Collect rays in bundles of 16
        mRays4[bundleIndex4].org_x[rayIndex4] = x;
        mRays4[bundleIndex4].org_y[rayIndex4] = y;
        mRays4[bundleIndex4].org_z[rayIndex4] = z;

        mRays4[bundleIndex4].dir_x[rayIndex4] = dir_x;
        mRays4[bundleIndex4].dir_y[rayIndex4] = dir_y;
        mRays4[bundleIndex4].dir_z[rayIndex4] = dir_z;

        mRays4[bundleIndex4].tnear[rayIndex4] = tNear;
        mRays4[bundleIndex4].tfar[rayIndex4] = tFar;
        mRays4[bundleIndex4].mask[rayIndex4] = mask;
        mRays4[bundleIndex4].flags[rayIndex4] = flag;

        // Set the validity of the ray in the bundle, -1 = Valied, 0 = Invalid
        mRays4Valid[bundleIndex4][rayIndex4] = -1;

        rayIndex8 = i % 8;

        if (rayIndex8 == 0)
            bundleIndex8++;

        // Collect rays in bundles of 16
        mRays8[bundleIndex8].org_x[rayIndex8] = x;
        mRays8[bundleIndex8].org_y[rayIndex8] = y;
        mRays8[bundleIndex8].org_z[rayIndex8] = z;

        mRays8[bundleIndex8].dir_x[rayIndex8] = dir_x;
        mRays8[bundleIndex8].dir_y[rayIndex8] = dir_y;
        mRays8[bundleIndex8].dir_z[rayIndex8] = dir_z;

        mRays8[bundleIndex8].tnear[rayIndex8] = tNear;
        mRays8[bundleIndex8].tfar[rayIndex8] = tFar;
        mRays8[bundleIndex8].mask[rayIndex8] = mask;
        mRays8[bundleIndex8].flags[rayIndex8] = flag;

        // Set the validity of the ray in the bundle, -1 = Valied, 0 = Invalid
        mRays8Valid[bundleIndex8][rayIndex8] = -1;

        rayIndex16 = i % 16;

        if (rayIndex16 == 0)
            bundleIndex16++;

        // Collect rays in bundles of 16
        mRays16[bundleIndex16].org_x[rayIndex16] = x;
        mRays16[bundleIndex16].org_y[rayIndex16] = y;
        mRays16[bundleIndex16].org_z[rayIndex16] = z;

        mRays16[bundleIndex16].dir_x[rayIndex16] = dir_x;
        mRays16[bundleIndex16].dir_y[rayIndex16] = dir_y;
        mRays16[bundleIndex16].dir_z[rayIndex16] = dir_z;

        mRays16[bundleIndex16].tnear[rayIndex16] = tNear;
        mRays16[bundleIndex16].tfar[rayIndex16] = tFar;
        mRays16[bundleIndex16].mask[rayIndex16] = mask;
        mRays16[bundleIndex16].flags[rayIndex16] = flag;

        // Set the validity of the ray in the bundle, -1 = Valied, 0 = Invalid
        mRays16Valid[bundleIndex16][rayIndex16] = -1;
    }
}

void EmbreeSolar::updateRay1Directions(std::vector<float> new_sun_vec)
{
    for (int i = 0; i < mRayCount; i++)
    {
        mRays[i].dir_x = new_sun_vec[0];
        mRays[i].dir_y = new_sun_vec[1];
        mRays[i].dir_z = new_sun_vec[2];
    }
}

void EmbreeSolar::updateRay4Directions(std::vector<float> new_sun_vec)
{
    for (int i = 0; i < mBundle4Count; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            if (mRays4Valid[i][j] == -1)
            {
                mRays4[i].dir_x[j] = new_sun_vec[0];
                mRays4[i].dir_y[j] = new_sun_vec[1];
                mRays4[i].dir_z[j] = new_sun_vec[2];
            }
        }
    }
}

void EmbreeSolar::updateRay8Directions(std::vector<float> new_sun_vec)
{
    for (int i = 0; i < mBundle8Count; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            if (mRays8Valid[i][j] == -1)
            {
                mRays8[i].dir_x[j] = new_sun_vec[0];
                mRays8[i].dir_y[j] = new_sun_vec[1];
                mRays8[i].dir_z[j] = new_sun_vec[2];
            }
        }
    }
}

void EmbreeSolar::updateRay16Directions(std::vector<float> new_sun_vec)
{
    for (int i = 0; i < mBundle16Count; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            if (mRays16Valid[i][j] == -1)
            {
                mRays16[i].dir_x[j] = new_sun_vec[0];
                mRays16[i].dir_y[j] = new_sun_vec[1];
                mRays16[i].dir_z[j] = new_sun_vec[2];
            }
        }
    }
}

void EmbreeSolar::raytrace_occ1()
{
    auto start = std::chrono::high_resolution_clock::now();
    printf("Testing --- rtcOccluded1 ---\n");
    int hitCounter = 0;
    for (int i = 0; i < mRayCount; i++)
    {
        RTCRay ray = mRays[i];
        rtcOccluded1(mScene, &ray);

        if (ray.tfar == -std::numeric_limits<float>::infinity())
            hitCounter++;
    }

    printf("Found %d intersections.\n", hitCounter);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Time elapsed: " << duration.count() << " seconds" << std::endl;
}

void EmbreeSolar::raytrace_occ4()
{
    auto start = std::chrono::high_resolution_clock::now();
    printf("Testing --- rtcOccluded4 ---\n");
    int hitCounter = 0;
    for (int i = 0; i < mBundle4Count; i++)
    {
        RTCRay4 rayBundle = mRays4[i];
        const int *valid = mRays4Valid[i];
        rtcOccluded4(valid, mScene, &rayBundle);

        for (int j = 0; j < 4; j++)
        {
            if (rayBundle.tfar[j] == -std::numeric_limits<float>::infinity())
                hitCounter++;
        }
    }

    printf("Found %d intersections.\n", hitCounter);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Time elapsed: " << duration.count() << " seconds" << std::endl;
}

void EmbreeSolar::raytrace_occ8()
{
    auto start = std::chrono::high_resolution_clock::now();
    printf("Testing --- rtcOccluded8 ---\n");
    int hitCounter = 0;
    for (int i = 0; i < mBundle8Count; i++)
    {
        RTCRay8 rayBundle = mRays8[i];
        const int *valid = mRays8Valid[i];
        rtcOccluded8(valid, mScene, &rayBundle);

        for (int j = 0; j < 8; j++)
        {
            if (rayBundle.tfar[j] == -std::numeric_limits<float>::infinity())
                hitCounter++;
        }
    }

    printf("Found %d intersections.\n", hitCounter);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration4 = end - start;
    std::cout << "Time elapsed: " << duration4.count() << " seconds" << std::endl;
}

void EmbreeSolar::raytrace_occ16()
{
    auto start = std::chrono::high_resolution_clock::now();
    printf("Testing --- rtcOccluded16 ---\n");
    int hitCounter = 0;
    for (int i = 0; i < mBundle16Count; i++)
    {
        RTCRay16 rayBundle = mRays16[i];
        const int *valid = mRays16Valid[i];
        rtcOccluded16(valid, mScene, &rayBundle);

        for (int j = 0; j < 16; j++)
        {
            if (rayBundle.tfar[j] == -std::numeric_limits<float>::infinity())
                hitCounter++;
        }
    }

    printf("Found %d intersections.\n", hitCounter);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration5 = end - start;
    std::cout << "Time elapsed: " << duration5.count() << " seconds" << std::endl;
}

void EmbreeSolar::raytrace_occ1_2(std::vector<int> &results, int &hitCounter)
{
    for (int i = 0; i < mRayCount; i++)
    {
        RTCRay ray = mRays[i];
        rtcOccluded1(mScene, &ray);

        if (ray.tfar == -std::numeric_limits<float>::infinity())
        {
            hitCounter++;
            results[i] = 1;
        }
    }
}

void EmbreeSolar::raytrace_occ4_2(std::vector<int> &results, int &hitCounter)
{
    for (int i = 0; i < mBundle4Count; i++)
    {
        RTCRay4 rayBundle = mRays4[i];
        const int *valid = mRays4Valid[i];
        rtcOccluded4(valid, mScene, &rayBundle);
        for (int j = 0; j < 4; j++)
        {
            int rayIndex = i * 8 + j;
            if (rayBundle.tfar[j] == -std::numeric_limits<float>::infinity())
            {
                hitCounter++;
                results[rayIndex] = 1;
            }
        }
    }
}

void EmbreeSolar::raytrace_occ8_2(std::vector<int> &results, int &hitCounter)
{
    for (int i = 0; i < mBundle8Count; i++)
    {
        RTCRay8 rayBundle = mRays8[i];
        const int *valid = mRays8Valid[i];
        rtcOccluded8(valid, mScene, &rayBundle);

        for (int j = 0; j < 8; j++)
        {
            int rayIndex = i * 8 + j;
            if (rayBundle.tfar[j] == -std::numeric_limits<float>::infinity())
            {
                hitCounter++;
                results[rayIndex] = 1;
            }
        }
    }
}

void EmbreeSolar::raytrace_occ16_2(std::vector<int> &results, int &hitCounter)
{
    for (int i = 0; i < mBundle16Count; i++)
    {
        RTCRay16 rayBundle = mRays16[i];
        const int *valid = mRays16Valid[i];
        rtcOccluded16(valid, mScene, &rayBundle);

        for (int j = 0; j < 16; j++)
        {
            int rayIndex = i * 16 + j;
            if (rayBundle.tfar[j] == -std::numeric_limits<float>::infinity())
            {
                hitCounter++;
                results[rayIndex] = 1;
            }
        }
    }
}

std::vector<std::vector<int>> EmbreeSolar::iterateRaytrace_occ1(std::vector<std::vector<float>> sun_vecs)
{
    // Define a 2D vector to store intersection results. Each postion is given the
    // initial values 0, which is changed to 1 if an intersection is found.
    auto all_results = std::vector<std::vector<int>>(sun_vecs.size(), std::vector<int>(mFaceCount, 0));
    int hitCounter = 0;
    auto start = std::chrono::high_resolution_clock::now();
    printf("Running rtcOccluded1 for %d sun vectors.\n", (int)sun_vecs.size());
    for (int i = 0; i < sun_vecs.size(); i++)
    {
        std::vector<int> &results = all_results[i];
        if (sun_vecs[i].size() == 3)
        {
            updateRay1Directions(sun_vecs[i]);
            raytrace_occ1_2(results, hitCounter);
        }
        else
            printf("Invalid sun vector size in EmbreeSolar::iterateRaytrace_occ1.\n");
    }

    printf("Found %d intersections.\n", hitCounter);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Time elapsed: " << duration.count() << " seconds" << std::endl;

    return all_results;
}

std::vector<std::vector<int>> EmbreeSolar::iterateRaytrace_occ4(std::vector<std::vector<float>> sun_vecs)
{
    // Define a 2D vector to store intersection results. Each postion is given the
    // initial values 0, which is changed to 1 if an intersection is found.
    auto all_results = std::vector<std::vector<int>>(sun_vecs.size(), std::vector<int>(mFaceCount, 0));
    int hitCounter = 0;
    auto start = std::chrono::high_resolution_clock::now();
    printf("Running rtcOccluded4 for %d sun vectors.\n", (int)sun_vecs.size());
    for (int i = 0; i < sun_vecs.size(); i++)
    {
        std::vector<int> &results = all_results[i];
        if (sun_vecs[i].size() == 3)
        {
            updateRay4Directions(sun_vecs[i]);
            raytrace_occ4_2(results, hitCounter);
        }
        else
            printf("Invalid sun vector size in EmbreeSolar::iterateRaytrace_occ4.\n");
    }

    printf("Found %d intersections.\n", hitCounter);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Time elapsed: " << duration.count() << " seconds" << std::endl;

    return all_results;
}

std::vector<std::vector<int>> EmbreeSolar::iterateRaytrace_occ8(std::vector<std::vector<float>> sun_vecs)
{
    // Define a 2D vector to store intersection results. Each postion is given the
    // initial values 0, which is changed to 1 if an intersection is found.
    auto all_results = std::vector<std::vector<int>>(sun_vecs.size(), std::vector<int>(mFaceCount, 0));
    auto start = std::chrono::high_resolution_clock::now();
    printf("Running rtcOccluded8 for %d sun vectors.\n", (int)sun_vecs.size());
    int hitCounter = 0;
    for (int i = 0; i < sun_vecs.size(); i++)
    {
        std::vector<int> &results = all_results[i];
        if (sun_vecs[i].size() == 3)
        {
            updateRay8Directions(sun_vecs[i]);
            raytrace_occ8_2(results, hitCounter);
        }
        else
            printf("Invalid sun vector size in EmbreeSolar::iterateRaytrace_occ8.\n");
    }

    printf("Found %d intersections.\n", hitCounter);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration4 = end - start;
    std::cout << "Time elapsed: " << duration4.count() << " seconds" << std::endl;
    return all_results;
}

std::vector<std::vector<int>> EmbreeSolar::iterateRaytrace_occ16(std::vector<std::vector<float>> sun_vecs)
{
    // Define a 2D vector to store intersection results. Each postion is given the
    // initial values 0, which is changed to 1 if an intersection is found.
    auto all_results = std::vector<std::vector<int>>(sun_vecs.size(), std::vector<int>(mFaceCount, 0));
    auto start = std::chrono::high_resolution_clock::now();
    printf("Running rtcOccluded16 for %d sun vectors.\n", (int)sun_vecs.size());
    int hitCounter = 0;
    for (int i = 0; i < sun_vecs.size(); i++)
    {
        std::vector<int> &results = all_results[i];
        if (sun_vecs[i].size() == 3)
        {
            updateRay16Directions(sun_vecs[i]);
            raytrace_occ16_2(results, hitCounter);
        }
        else
            printf("Invalid sun vector size in EmbreeSolar::iterateRaytrace_occ16.\n");
    }

    printf("Found %d intersections.\n", hitCounter);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration4 = end - start;
    std::cout << "Time elapsed: " << duration4.count() << " seconds" << std::endl;
    return all_results;
}

namespace py = pybind11;

PYBIND11_MODULE(py_embree_solar, m)
{
    py::class_<EmbreeSolar>(m, "PyEmbreeSolar")
        .def(py::init<>())
        .def(py::init<std::vector<std::vector<float>>, std::vector<std::vector<int>>, std::vector<float>>())
        .def("createDevice", &EmbreeSolar::createDevice)
        .def("createScene", &EmbreeSolar::createScene)
        .def("createGeomPlane", &EmbreeSolar::createGeomPlane)
        .def("createGridRays", &EmbreeSolar::createGridRays)
        .def("bundleRays", &EmbreeSolar::bundleRays)
        .def("raytrace_occ1", &EmbreeSolar::raytrace_occ1)
        .def("raytrace_occ4", &EmbreeSolar::raytrace_occ4)
        .def("raytrace_occ8", &EmbreeSolar::raytrace_occ8)
        .def("raytrace_occ16", &EmbreeSolar::raytrace_occ16)
        .def("updateRay1Directions", &EmbreeSolar::updateRay1Directions)
        .def("updateRay4Directions", &EmbreeSolar::updateRay4Directions)
        .def("updateRay8Directions", &EmbreeSolar::updateRay8Directions)
        .def("updateRay16Directions", &EmbreeSolar::updateRay16Directions)
        .def("iterateRaytrace_occ1", [](EmbreeSolar &self, std::vector<std::vector<float>> sun_vecs)
             { py::array out = py::cast(self.iterateRaytrace_occ1(sun_vecs)); return out; })
        .def("iterateRaytrace_occ4", [](EmbreeSolar &self, std::vector<std::vector<float>> sun_vecs)
             { py::array out = py::cast(self.iterateRaytrace_occ4(sun_vecs)); return out; })
        .def("iterateRaytrace_occ8", [](EmbreeSolar &self, std::vector<std::vector<float>> sun_vecs)
             { py::array out = py::cast(self.iterateRaytrace_occ8(sun_vecs)); return out; })
        .def("iterateRaytrace_occ16", [](EmbreeSolar &self, std::vector<std::vector<float>> sun_vecs)
             { py::array out = py::cast(self.iterateRaytrace_occ16(sun_vecs)); return out; });
}
