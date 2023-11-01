#include "embree_solar.h"

EmbreeSolar::EmbreeSolar()
{
    printf("Creating EmbreeSolar instance with default constructor.\n");

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

    std::cout << "Model setup with plane geometry contais:" << std::endl;
    std::cout << "Number of vertices: " << mVertexCount << std::endl;
    std::cout << "Number of faces: " << mFaceCount << std::endl;

    mSkydome = new Skydome();
    mSunrays = new Sunrays();

    CreateDevice();
    CreateScene();
    CreateGeomPlane();
    CalcFaceMidPoints();

    std::cout << "Model setup with predefinde settings complete." << std::endl;
}

EmbreeSolar::EmbreeSolar(std::vector<std::vector<float>> vertices, std::vector<std::vector<int>> faces)
{
    printf("Creating EmbreeSolar instance with custom geometry.\n");

    mVertexCount = vertices.size();
    mFaceCount = faces.size();

    CreateDevice();
    CreateScene();
    CreateGeom(vertices, faces);
    CalcFaceMidPoints();

    mSkydome = new Skydome(10);
    mSunrays = new Sunrays(mFaceMidPts, mFaceCount);

    std::cout << "Model setup with mesh geometry complete." << std::endl;
}

EmbreeSolar::~EmbreeSolar()
{
    delete mSkydome;
    delete mSunrays;

    delete[] mFaceMidPts;
    delete[] mFaces;
    delete[] mVertices;

    rtcReleaseScene(mScene);
    rtcReleaseDevice(mDevice);

    printf("Destructor called.\n");
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

void EmbreeSolar::CreateDevice()
{
    mDevice = rtcNewDevice(NULL);

    if (!mDevice)
        printf("error %d: cannot create device\n", rtcGetDeviceError(NULL));

    // rtcSetDeviceErrorFunction(mDevice, errorFunction, NULL);

    printf("Device created.\n");
}

void EmbreeSolar::ErrorFunction(void *userPtr, enum RTCError error, const char *str)
{
    printf("error %d: %s\n", error, str);
}

void EmbreeSolar::CreateScene()
{
    mScene = rtcNewScene(mDevice);

    printf("Scene created.\n");
}

void EmbreeSolar::CreateGeom(std::vector<std::vector<float>> vertices, std::vector<std::vector<int>> faces)
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
    unsigned int geomID = rtcAttachGeometry(mScene, mGeometry);
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

void EmbreeSolar::Raytrace_occ1(std::vector<int> &results, int &hitCounter)
{
    int nRays = mSunrays->GetRayCount();
    for (int i = 0; i < nRays; i++)
    {
        RTCRay ray = mSunrays->GetRays()[i];
        rtcOccluded1(mScene, &ray);

        if (ray.tfar == -std::numeric_limits<float>::infinity())
        {
            hitCounter++;
            results[i] = 1;
        }
    }
}

void EmbreeSolar::Raytrace_occ4(std::vector<int> &results, int &hitCounter)
{
    int nBundles = mSunrays->GetBundle4Count();
    for (int i = 0; i < nBundles; i++)
    {
        RTCRay4 rayBundle = mSunrays->GetRays4()[i];
        const int *valid = mSunrays->GetValid4()[i];
        rtcOccluded4(valid, mScene, &rayBundle);
        for (int j = 0; j < 4; j++)
        {
            int rayIndex = i * 4 + j;
            if (rayBundle.tfar[j] == -std::numeric_limits<float>::infinity())
            {
                hitCounter++;
                results[rayIndex] = 1;
            }
        }
    }
}

void EmbreeSolar::Raytrace_occ8(std::vector<int> &results, int &hitCounter)
{
    int nBundles = mSunrays->GetBundle8Count();
    for (int i = 0; i < nBundles; i++)
    {
        RTCRay8 rayBundle = mSunrays->GetRays8()[i];
        const int *valid = mSunrays->GetValid8()[i];
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

void EmbreeSolar::Raytrace_occ16(std::vector<int> &results, int &hitCounter)
{
    int nBundles = mSunrays->GetBundle16Count();
    for (int i = 0; i < nBundles; i++)
    {
        RTCRay16 rayBundle = mSunrays->GetRays16()[i];
        const int *valid = mSunrays->GetValid16()[i];
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

std::vector<std::vector<int>> EmbreeSolar::SunRaytrace_Occ1(std::vector<std::vector<float>> sun_vecs)
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
            mSunrays->UpdateRay1Directions(sun_vecs[i]);
            Raytrace_occ1(results, hitCounter);
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

std::vector<std::vector<int>> EmbreeSolar::SunRaytrace_Occ4(std::vector<std::vector<float>> sun_vecs)
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
            mSunrays->UpdateRay4Directions(sun_vecs[i]);
            Raytrace_occ4(results, hitCounter);
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

std::vector<std::vector<int>> EmbreeSolar::SunRaytrace_Occ8(std::vector<std::vector<float>> sun_vecs)
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
            mSunrays->UpdateRay8Directions(sun_vecs[i]);
            Raytrace_occ8(results, hitCounter);
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

std::vector<std::vector<int>> EmbreeSolar::SunRaytrace_Occ16(std::vector<std::vector<float>> sun_vecs)
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
            mSunrays->UpdateRay16Directions(sun_vecs[i]);
            Raytrace_occ16(results, hitCounter);
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

std::vector<float> EmbreeSolar::SkyRaytrace_Occ1()
{
    int hitCounter = 0;
    float hitPortion = 0.0f;
    // Compute diffuse sky portion by iterating over all faces in the mesh
    auto all_results = std::vector<float>(mFaceCount, 0);
    auto start = std::chrono::high_resolution_clock::now();
    printf("Running diffuse rtcOccluded1 for sun vectors.\n");
    for (int i = 0; i < mFaceCount; i++)
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
            }
        }
        all_results[i] = hitPortion;
        printf("Hit portion for face %d is: %f.\n", i, hitPortion);
    }

    printf("Found %d intersections.\n", hitCounter);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Time elapsed: " << duration.count() << " seconds" << std::endl;

    return all_results;
}

std::vector<float> EmbreeSolar::SkyRaytrace_Occ4()
{
    int hitCounter = 0;
    int intAttempts = 0;
    float hitPortion = 0.0f;
    // Compute diffuse sky portion by iterating over all faces in the mesh
    auto all_results = std::vector<float>(mFaceCount, 0);
    auto start = std::chrono::high_resolution_clock::now();
    printf("Running diffuse rtcOccluded4 for sun vectors.\n");
    for (int i = 0; i < mFaceCount; i++)
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
                }
                intAttempts++;
            }
        }
        all_results[i] = hitPortion;
        printf("Hit portion for face %d is: %f.\n", i, hitPortion);
    }

    printf("Found %d intersections in %d attempts.\n", hitCounter, intAttempts);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Time elapsed: " << duration.count() << " seconds" << std::endl;

    return all_results;
}

std::vector<float> EmbreeSolar::SkyRaytrace_Occ8()
{
    int hitCounter = 0;
    int intAttempts = 0;
    float hitPortion = 0.0f;
    // Compute diffuse sky portion by iterating over all faces in the mesh
    auto all_results = std::vector<float>(mFaceCount, 0);
    auto start = std::chrono::high_resolution_clock::now();
    printf("Running diffuse rtcOccluded8 for sun vectors.\n");
    for (int i = 0; i < mFaceCount; i++)
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
                }
                intAttempts++;
            }
        }
        all_results[i] = hitPortion;
        printf("Hit portion for face %d is: %f.\n", i, hitPortion);
    }

    printf("Found %d intersections in %d attempts.\n", hitCounter, intAttempts);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Time elapsed: " << duration.count() << " seconds" << std::endl;

    return all_results;
}

std::vector<float> EmbreeSolar::SkyRaytrace_Occ16()
{
    int hitCounter = 0;
    int intAttempts = 0;
    float hitPortion = 0.0f;
    // Compute diffuse sky portion by iterating over all faces in the mesh
    auto all_results = std::vector<float>(mFaceCount, 0);
    auto start = std::chrono::high_resolution_clock::now();
    printf("Running diffuse rtcOccluded16 for sun vectors.\n");
    for (int i = 0; i < mFaceCount; i++)
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
                }
                intAttempts++;
            }
        }
        all_results[i] = hitPortion;
        printf("Hit portion for face %d is: %f.\n", i, hitPortion);
    }

    printf("Found %d intersections in %d attempts.\n", hitCounter, intAttempts);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end - start;
    std::cout << "Time elapsed: " << duration.count() << " seconds" << std::endl;

    return all_results;
}

#ifdef PYTHON_MODULE

namespace py = pybind11;

PYBIND11_MODULE(py_embree_solar, m)
{
    py::class_<EmbreeSolar>(m, "PyEmbreeSolar")
        .def(py::init<>())
        .def(py::init<std::vector<std::vector<float>>, std::vector<std::vector<int>>>())
        .def("getSkydomeFaces", [](EmbreeSolar &self)
             { py::array out = py::cast(self.GetSkydomeFaces()); return out; })
        .def("getSkydomeVertices", [](EmbreeSolar &self)
             { py::array out = py::cast(self.GetSkydomeVertices()); return out; })
        .def("getSkydomeRays", [](EmbreeSolar &self)
             { py::array out = py::cast(self.GetSkydomeRayDirections()); return out; })
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
             { py::array out = py::cast(self.SkyRaytrace_Occ16()); return out; });
}

#endif