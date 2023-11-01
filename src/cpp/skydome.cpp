#include "skydome.h"

Skydome::Skydome()
{
    printf("Skydome created with default constructor.\n");
    mRayOrigin = {0.0, 0.0, 0.0};
    CreateMesh(10);
    InitRays(mRayDirections.size());
    CreateRays();
    BundleRays();

    printf("Skydome instance created, ready for raytracing.\n");
}

Skydome::Skydome(int nStrips)
{
    mRayOrigin = {0.0, 0.0, 0.0};
    CreateMesh(nStrips);
    InitRays(mRayDirections.size());
    CreateRays();
    BundleRays();

    printf("Skydome instance created, ready for raytracing.\n");
}

Skydome::~Skydome()
{
    delete[] mRays;
    delete[] mRays4;
    delete[] mRays8;
    delete[] mRays16;

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

    std::cout << "Skydome rays data: " << std::endl;
    std::cout << "Number of rays: " << mRayCount << std::endl;
    std::cout << "Number of 4 bundles: " << mBundle4Count << std::endl;
    std::cout << "Number of 8 bundles: " << mBundle8Count << std::endl;
    std::cout << "Number of 16 bundles: " << mBundle16Count << std::endl;

    // Defining the arrays for the rays on the heap
    mRays = new RTCRay[mRayCount];
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

RTCRay *Skydome::GetRays()
{
    return mRays;
}

RTCRay4 *Skydome::GetRays4()
{
    return mRays4;
}

RTCRay8 *Skydome::GetRays8()
{
    return mRays8;
}

RTCRay16 *Skydome::GetRays16()
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
    mRays = new RTCRay[mRayCount];

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

void Skydome::CreateMesh(int nStrips)
{
    int topCapDiv = 4;
    float maxAzim = 2 * M_PI;
    float maxElev = 0.5 * M_PI;
    float dElev = maxElev / nStrips;

    float elev = maxElev - dElev;
    float azim = 0;

    float topCapArea = CalcSphereCapArea(elev);
    float hemisphereArea = CalcHemisphereArea();
    float targetArea = topCapArea / topCapDiv;
    float faceAreaPart = targetArea / hemisphereArea;

    // Start by adding the 4 rays for the 4 top cap quads
    GetTopCap(maxElev, elev, maxAzim, topCapDiv, faceAreaPart);

    for (int i = 0; i < nStrips - 1; ++i)
    {
        azim = 0;
        float nextElev = elev - dElev;
        float stripArea = CalcSphereStripArea(elev, nextElev);
        int nAzim = (int)(stripArea / targetArea);
        float dAzim = maxAzim / nAzim;
        faceAreaPart = (stripArea / nAzim) / hemisphereArea;
        float midElev = (elev + nextElev) / 2.0;

        for (int j = 0; j < nAzim; ++j)
        {
            float nextAzim = azim + dAzim;
            CreateDomeMeshQuad(azim, nextAzim, elev, nextElev);

            float midAzim = (azim + nextAzim) / 2.0;
            float x = cos(midElev) * cos(midAzim);
            float y = cos(midElev) * sin(midAzim);
            float z = sin(midElev);

            std::vector<float> rayDir = {x, y, z};
            mRayDirections.push_back(rayDir);
            mRayAreas.push_back(faceAreaPart);
            azim += dAzim;
        }
        elev = elev - dElev;
    }
}

void Skydome::GetTopCap(float maxElev, float elev, float maxAzim, int nAzim, float faceAreaPart)
{
    float elev_mid = (elev + maxElev) / 2.0;
    float dAzim = maxAzim / nAzim;
    float azim_mid = dAzim / 2.0;
    float azim = 0;

    for (int i = 0; i < nAzim; ++i)
    {
        float x = cos(elev_mid) * cos(azim_mid);
        float y = cos(elev_mid) * sin(azim_mid);
        float z = sin(elev_mid);

        std::vector<float> rayDir = {x, y, z};
        mRayDirections.push_back(rayDir);
        mRayAreas.push_back(faceAreaPart);

        CreateTopCapQuads(maxElev, elev, azim, dAzim);

        azim = azim + dAzim;
        azim_mid = azim_mid + dAzim;
    }
}

float Skydome::CalcHemisphereArea()
{
    float r = 1.0f;
    float area = 2 * M_PI * r * r;
    return area;
}

float Skydome::CalcSphereCapArea(float elevation)
{
    float r = 1.0;
    float h = r - r * sin(elevation);
    float C = 2 * sqrt(h * (2 * r - h));
    float area = M_PI * (((C * C) / 4) + h * h);
    return area;
}

// Create a quad from 4 points
void Skydome::CreateDomeMeshQuad(float azim, float nextAzim, float elev, float nextElev)
{
    float rayLength = 1.0;

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

    int index0 = mVertices.size() - 4;
    int index1 = mVertices.size() - 3;
    int index2 = mVertices.size() - 2;
    int index3 = mVertices.size() - 1;

    std::vector<int> face1 = {index0, index1, index2};
    std::vector<int> face2 = {index1, index3, index2};

    mFaces.push_back(face1);
    mFaces.push_back(face2);
}

// Create the top 4 quads that close the dome at the top.
void Skydome::CreateTopCapQuads(float elev, float nextElev, float azim, float dAzim)
{
    float rayLength = 1.0f;

    float midAzim = azim + (dAzim / 2);
    float nextAzim = azim + dAzim;

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

    float x3 = rayLength * cos(nextElev) * cos(midAzim);
    float y3 = rayLength * cos(nextElev) * sin(midAzim);
    float z3 = rayLength * sin(nextElev);
    std::vector<float> pt3 = {x3, y3, z3};
    mVertices.push_back(pt3);

    float x4 = rayLength * cos(nextElev) * cos(nextAzim);
    float y4 = rayLength * cos(nextElev) * sin(nextAzim);
    float z4 = rayLength * sin(nextElev);
    std::vector<float> pt4 = {x4, y4, z4};
    mVertices.push_back(pt4);

    int index0 = mVertices.size() - 4;
    int index1 = mVertices.size() - 3;
    int index2 = mVertices.size() - 2;
    int index3 = mVertices.size() - 1;

    std::vector<int> face1 = {index0, index1, index2};
    std::vector<int> face2 = {index2, index3, index0};

    mFaces.push_back(face1);
    mFaces.push_back(face2);
}

// Ref: https://www.easycalculation.com/shapes/learn-spherical-cap.php
float Skydome::CalcSphereStripArea(float elev1, float elev2)
{
    float r = 1.0f;
    float h1 = r - r * sin(elev1);
    float h2 = r - r * sin(elev2);
    float C1 = 2 * sqrt(h1 * (2 * r - h1));
    float C2 = 2 * sqrt(h2 * (2 * r - h2));
    float area1 = M_PI * (((C1 * C1) / 4) + h1 * h1);
    float area2 = M_PI * (((C2 * C2) / 4) + h2 * h2);
    return area2 - area1;
}
