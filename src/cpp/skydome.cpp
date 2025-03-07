#include "skydome.h"

Skydome::Skydome()
{
    mRayOrigin = {0.0, 0.0, 0.0};
    CreateEqualAreaMesh(10);
    InitRays((int)mRayDirections.size());
    CreateRays();
    BundleRays();

    info("Skydome instance created with default constructor, ready for raytracing.");
}

Skydome::Skydome(int skyType)
{
    mRayOrigin = {0.0, 0.0, 0.0};
    if (skyType == 1)
    {
        CreateEqualAreaMesh(10);
    }
    else if (skyType == 2)
    {
        CreateTregenzaMesh();
    }
    else if (skyType == 3)
    {
        CreateReinhartMesh();
    }
    else
    {
        warning("Invalid sky type, defaulting to equal area mesh.");
        CreateEqualAreaMesh(10);
    }
    InitRays((int)mRayDirections.size());
    CreateRays();
    BundleRays();

    info("Skydome instance created, ready for raytracing.");
}

Skydome::~Skydome()
{
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
}

void Skydome::InitRays(int rayCount)
{
    mRayCount = rayCount;
    mBundle4Count = ceil((float)mRayCount / 4.0f);
    mBundle8Count = ceil((float)mRayCount / 8.0f);
    mBundle16Count = ceil((float)mRayCount / 16.0f);

    debug("Skydome rays data:");
    debug("Number of rays:" + str(mRayCount) + ".");
    debug("Number of 4 bundles:" + str(mBundle4Count) + ".");
    debug("Number of 8 bundles:" + str(mBundle8Count) + ".");
    debug("Number of 16 bundles:" + str(mBundle16Count) + ".");

    mRays = std::vector<RTCRay>(mRayCount);
    mRays4 = std::vector<RTCRay4>(mBundle4Count);
    mRays8 = std::vector<RTCRay8>(mBundle8Count);
    mRays16 = std::vector<RTCRay16>(mBundle16Count);

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

int Skydome::GetFaceCount()
{
    return (int)mFaces.size();
}

int Skydome::GetRayCount()
{
    return mRayCount;
}

int Skydome::GetBundle4Count()
{
    return mBundle4Count;
}

int Skydome::GetBundle8Count()
{
    return mBundle8Count;
}

int Skydome::GetBundle16Count()
{
    return mBundle16Count;
}

std::vector<RTCRay> &Skydome::GetRays()
{
    return mRays;
}

std::vector<RTCRay4> &Skydome::GetRays4()
{
    return mRays4;
}

std::vector<RTCRay8> &Skydome::GetRays8()
{
    return mRays8;
}

std::vector<RTCRay16> &Skydome::GetRays16()
{
    return mRays16;
}

int **Skydome::GetValid4()
{
    return mRays4Valid;
}

int **Skydome::GetValid8()
{
    return mRays8Valid;
}

int **Skydome::GetValid16()
{
    return mRays16Valid;
}

std::vector<std::vector<int>> Skydome::GetFaces()
{
    return mFaces;
}

std::vector<std::vector<float>> Skydome::GetVertices()
{
    return mVertices;
}

std::vector<std::vector<float>> Skydome::GetRayDirections()
{
    return mRayDirections;
}

std::vector<float> Skydome::GetRayAreas()
{
    return mRayAreas;
}

void Skydome::TranslateRays(Vertex new_origin)
{
    for (int i = 0; i < mRayCount; i++)
    {
        mRays[i].org_x = new_origin.x;
        mRays[i].org_y = new_origin.y;
        mRays[i].org_z = new_origin.z;
    }
}

void Skydome::Translate4Rays(Vertex new_origin)
{
    for (int i = 0; i < mBundle4Count; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            if (mRays4Valid[i][j] == -1)
            {
                mRays4[i].org_x[j] = new_origin.x;
                mRays4[i].org_y[j] = new_origin.y;
                mRays4[i].org_z[j] = new_origin.z;
            }
        }
    }
}

void Skydome::Translate8Rays(Vertex new_origin)
{
    for (int i = 0; i < mBundle8Count; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            if (mRays8Valid[i][j] == -1)
            {
                mRays8[i].org_x[j] = new_origin.x;
                mRays8[i].org_y[j] = new_origin.y;
                mRays8[i].org_z[j] = new_origin.z;
            }
        }
    }
}

void Skydome::Translate16Rays(Vertex new_origin)
{
    for (int i = 0; i < mBundle16Count; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            if (mRays16Valid[i][j] == -1)
            {
                mRays16[i].org_x[j] = new_origin.x;
                mRays16[i].org_y[j] = new_origin.y;
                mRays16[i].org_z[j] = new_origin.z;
            }
        }
    }
}

void Skydome::BundleRays()
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

void Skydome::CreateRays()
{
    for (int i = 0; i < mRayCount; i++)
    {
        RTCRay ray;
        ray.org_x = mRayOrigin[0];
        ray.org_y = mRayOrigin[1];
        ray.org_z = mRayOrigin[2];
        ray.dir_x = mRayDirections[i][0];
        ray.dir_y = mRayDirections[i][1];
        ray.dir_z = mRayDirections[i][2];
        ray.tnear = 0.05f;
        ray.tfar = std::numeric_limits<float>::infinity();
        ray.mask = 0xFFFFFFFF;
        ray.time = 0.0f;
        mRays[i] = ray;
    }
}

void Skydome::CreateEqualAreaMesh(int nStrips)
{
    int topCapDiv = 4;
    float maxAzim = 2.0f * M_PI;
    float maxElev = 0.5f * M_PI;
    float dElev = maxElev / nStrips;

    float elev = maxElev - dElev;
    float azim = 0.0f;

    float topCapArea = CalcSphereCapArea(elev);
    float domeArea = CalcHemisphereArea();
    float targetArea = topCapArea / topCapDiv;
    float faceAreaPart = targetArea / domeArea;

    // Start by adding the 4 rays for the 4 top cap quads
    GetTopCap(maxElev, elev, maxAzim, topCapDiv, faceAreaPart);

    for (int i = 0; i < nStrips - 1; ++i)
    {
        azim = 0.0f;
        float nextElev = elev - dElev;
        float stripArea = CalcSphereStripArea(elev, nextElev);
        int nAzim = (int)(stripArea / targetArea);
        float dAzim = maxAzim / nAzim;
        faceAreaPart = (stripArea / nAzim) / domeArea;
        float midElev = (elev + nextElev) / 2.0f;

        for (int j = 0; j < nAzim; ++j)
        {
            float nextAzim = azim + dAzim;
            float midAzim = (azim + nextAzim) / 2.0f;
            CreateMeshQuad(azim, nextAzim, elev, nextElev);
            std::vector<float> rayDir = Spherical2Cartesian(1.0, midElev, midAzim);
            mRayDirections.push_back(rayDir);
            mRayAreas.push_back(faceAreaPart);
            azim += dAzim;
        }
        elev = elev - dElev;
    }

    info("Equal area skydome mesh created.");
}

void Skydome::CreateTregenzaMesh()
{
    // Elevation steps in degrees
    const std::vector<float> elevs = {0.0, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0};
    const std::vector<int> patchCounts = {30, 30, 24, 24, 18, 12, 6, 1};

    const float elevStep = Deg2Rad(12.0);
    const int bands = 8; // 7 bands + zenith patch
    const float domeArea = CalcHemisphereArea();

    for (int i = 0; i < bands; i++)
    {
        float patchCount = patchCounts[i];
        float elev = Deg2Rad(elevs[i]);
        float elevNext = elev + elevStep;
        float azimStep = 2 * M_PI / patchCount;

        if (elev != Deg2Rad(84.0))
        {
            for (int j = 0; j < patchCount; ++j)
            {
                float azim = j * azimStep;
                float azimNext = (j + 1) * azimStep;
                float midElev = (elev + elevNext) / 2.0;
                float midAzim = (azim + azimNext) / 2.0f;
                CreateMeshQuad(azim, azimNext, elev, elevNext);

                std::vector<float> rayDir = Spherical2Cartesian(1.0, midElev, midAzim);
                mRayDirections.push_back(rayDir);

                float patchArea = CalcSpherePatchArea(1.0, elev, elevNext, azim, azimNext);
                mRayAreas.push_back(patchArea / domeArea);
            }
        }
        else
        {
            CreateTregenzaZenithPatch(elev);
        }
    }

    info("Tregenza skydome mesh created.");
}

void Skydome::CreateReinhartMesh()
{
    // Elevation steps in degrees
    const std::vector<float> elevs = {0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0, 54.0, 60.0, 66.0, 72.0, 78.0, 84.0};
    const std::vector<int> patchCounts = {60, 60, 60, 60, 48, 48, 48, 48, 36, 36, 24, 24, 12, 12, 4};

    const float domeArea = CalcHemisphereArea();
    const float elevStep = Deg2Rad(6.0);
    const int bands = 15; // 14 bands + zenith patch

    for (int i = 0; i < bands; i++)
    {
        float patchCount = patchCounts[i];
        float elev = Deg2Rad(elevs[i]);
        float elevNext = elev + elevStep;
        float azimStep = 2 * M_PI / patchCount;

        if (elev != Deg2Rad(84.0))
        {
            for (int j = 0; j < patchCount; ++j)
            {
                float azim = j * azimStep;
                float azimNext = (j + 1) * azimStep;
                float midElev = (elev + elevNext) / 2.0;
                float midAzim = (azim + azimNext) / 2.0f;

                CreateMeshQuad(azim, azimNext, elev, elevNext);
                std::vector<float> rayDir = Spherical2Cartesian(1.0, midElev, midAzim);
                mRayDirections.push_back(rayDir);

                float patchArea = CalcSpherePatchArea(1.0, elev, elevNext, azim, azimNext);
                patchArea = patchArea / domeArea;
                mRayAreas.push_back(patchArea);
            }
        }
        else
        {
            CreateReinhartZenithPatch(elev);
        }
    }

    info("Reinhart skydome mesh created.");
}

void Skydome::CreateEqualAreaZenithPatch(float elev, float nextElev, float azim, float dAzim)
{
    // Create the top 4 quads that close the dome at the top.

    float rayLength = 1.0f;

    float midAzim = azim + (dAzim / 2);
    float nextAzim = azim + dAzim;

    std::vector<float> pt1 = Spherical2Cartesian(rayLength, elev, azim);
    std::vector<float> pt2 = Spherical2Cartesian(1.0, nextElev, azim);
    std::vector<float> pt3 = Spherical2Cartesian(1.0, nextElev, midAzim);
    std::vector<float> pt4 = Spherical2Cartesian(rayLength, nextElev, nextAzim);

    mVertices.push_back(pt1);
    mVertices.push_back(pt2);
    mVertices.push_back(pt3);
    mVertices.push_back(pt4);

    int vCount = (int)mVertices.size();

    int index0 = vCount - 4;
    int index1 = vCount - 3;
    int index2 = vCount - 2;
    int index3 = vCount - 1;

    std::vector<int> face1 = {index0, index1, index2};
    std::vector<int> face2 = {index2, index3, index0};

    mFaces.push_back(face1);
    mFaces.push_back(face2);
}

void Skydome::CreateTregenzaZenithPatch(float elevation)
{
    const int vCount = (int)mVertices.size();
    const float domeArea = CalcHemisphereArea();

    // Create ray for zenith patch center
    std::vector<float> rayDir = {0, 0, 1.0};
    mRayDirections.push_back(rayDir);

    // Add zenith patch area
    float capArea = CalcSphereCapArea(elevation);
    mRayAreas.push_back(capArea / domeArea);

    for (int i = 0; i < 6; i++)
    {
        float azimuth_step = 2 * M_PI / 6;
        float azim = i * azimuth_step;
        mVertices.push_back(Spherical2Cartesian(1.0, elevation, azim));

        int idx1 = vCount + i;
        int idx2 = vCount + ((i + 1) % 6);
        int idx3 = vCount + 6; // Zenith point

        std::vector<int> face1 = {idx1, idx2, idx3};
        mFaces.push_back(face1);
    }

    std::vector<float> zenith = {0.0, 0.0, 1.0};
    mVertices.push_back(zenith);
}

void Skydome::CreateReinhartZenithPatch(float elevation)
{
    const int vCount = (int)mVertices.size();
    const int idxZenith = vCount + 12;
    const float capArea = CalcSphereCapArea(elevation);
    const float domeArea = CalcHemisphereArea();

    // Splitting the zenith patch into 4 quarters
    for (int i = 0; i < 4; i++)
    {
        float azimuth_step = M_PI / 2.0;
        float azim = i * azimuth_step;
        float azim_next = (i + 1) * azimuth_step;

        std::vector<float> pt1 = Spherical2Cartesian(1.0, elevation, azim);
        std::vector<float> pt2 = Spherical2Cartesian(1.0, elevation, azim + M_PI / 6.0);
        std::vector<float> pt3 = Spherical2Cartesian(1.0, elevation, azim + (2.0 * M_PI / 6.0));

        int idx1 = vCount + i * 4;
        int idx2 = vCount + i * 4 + 1;
        int idx3 = vCount + i * 4 + 2;
        int idx4 = vCount + (i * 4 + 3) % 12;

        mVertices.push_back(pt1);
        mVertices.push_back(pt2);
        mVertices.push_back(pt3);

        std::vector<int> face1 = {idx1, idx2, idxZenith};
        std::vector<int> face2 = {idx2, idx3, idxZenith};
        std::vector<int> face3 = {idx3, idx4, idxZenith};

        mFaces.push_back(face1);
        mFaces.push_back(face2);
        mFaces.push_back(face3);

        // Create ray for zenith quarter center
        float midAzim = (azim + azim_next) / 2.0;
        float midElev = (elevation + M_PI / 2.0) / 2.0;
        std::vector<float> rayDir = Spherical2Cartesian(1.0, midElev, midAzim);
        mRayDirections.push_back(rayDir);
        float quarterArea = capArea / 4.0;
        mRayAreas.push_back(quarterArea / domeArea);
    }

    // Add the zenith point
    std::vector<float> zenith = {0.0, 0.0, 1.0};
    mVertices.push_back(zenith);
}

void Skydome::GetTopCap(float maxElev, float elev, float maxAzim, int nAzim, float faceAreaPart)
{
    float elev_mid = (elev + maxElev) / 2.0f;
    float dAzim = maxAzim / nAzim;
    float azim_mid = dAzim / 2.0f;
    float azim = 0.0f;

    for (int i = 0; i < nAzim; ++i)
    {
        float x = cos(elev_mid) * cos(azim_mid);
        float y = cos(elev_mid) * sin(azim_mid);
        float z = sin(elev_mid);

        std::vector<float> rayDir = {x, y, z};
        mRayDirections.push_back(rayDir);
        mRayAreas.push_back(faceAreaPart);

        CreateEqualAreaZenithPatch(maxElev, elev, azim, dAzim);

        azim = azim + dAzim;
        azim_mid = azim_mid + dAzim;
    }
}

float Skydome::CalcHemisphereArea()
{
    float r = 1.0f;
    float area = 2.0f * M_PI * r * r;
    return area;
}

float Skydome::CalcSphereCapArea(float elevation)
{
    float polar_angle = M_PI / 2.0 - elevation;
    float r = 1.0;
    float area = 2.0 * M_PI * r * r * (1 - cos(polar_angle));
    return area;
}

void Skydome::CreateMeshQuad(float azim, float nextAzim, float elev, float nextElev)
{
    // Create a quad from 4 points
    float rayLength = 1.0f;

    float x1 = rayLength * cos(elev) * cos(azim);
    float y1 = rayLength * cos(elev) * sin(azim);
    float z1 = rayLength * sin(elev);
    std::vector<float> pt1 = {x1, y1, z1};
    mVertices.push_back(pt1);

    float x2 = rayLength * cos(nextElev) * cos(azim);
    float y2 = rayLength * cos(nextElev) * sin(azim);
    float z2 = rayLength * sin(nextElev);
    std::vector<float> pt2 = {x2, y2, z2};
    mVertices.push_back(pt2);

    float x3 = rayLength * cos(elev) * cos(nextAzim);
    float y3 = rayLength * cos(elev) * sin(nextAzim);
    float z3 = rayLength * sin(elev);
    std::vector<float> pt3 = {x3, y3, z3};
    mVertices.push_back(pt3);

    float x4 = rayLength * cos(nextElev) * cos(nextAzim);
    float y4 = rayLength * cos(nextElev) * sin(nextAzim);
    float z4 = rayLength * sin(nextElev);
    std::vector<float> pt4 = {x4, y4, z4};
    mVertices.push_back(pt4);

    int vCount = (int)mVertices.size();

    int index0 = vCount - 4;
    int index1 = vCount - 3;
    int index2 = vCount - 2;
    int index3 = vCount - 1;

    std::vector<int> face1 = {index0, index1, index2};
    std::vector<int> face2 = {index1, index3, index2};

    mFaces.push_back(face1);
    mFaces.push_back(face2);
}

float Skydome::CalcSphereStripArea(float elev1, float elev2)
{
    // Ref: https://www.easycalculation.com/shapes/learn-spherical-cap.php

    float r = 1.0f;
    float h1 = r - r * sin(elev1);
    float h2 = r - r * sin(elev2);
    float C1 = 2.0f * sqrt(h1 * (2.0f * r - h1));
    float C2 = 2.0f * sqrt(h2 * (2.0f * r - h2));
    float area1 = M_PI * (((C1 * C1) / 4.0f) + h1 * h1);
    float area2 = M_PI * (((C2 * C2) / 4.0f) + h2 * h2);
    return area2 - area1;
}

float Skydome::CalcSpherePatchArea(float r, float elev1, float elev2, float azim1, float azim2)
{
    double area = r * r * std::abs((azim2 - azim1) * (std::sin(elev2) - std::sin(elev1)));
    return area;
}