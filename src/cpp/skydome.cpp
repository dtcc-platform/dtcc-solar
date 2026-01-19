#include "skydome.h"

Skydome::Skydome()
{
    mRayOrigin = {0.0, 0.0, 0.0};
    CreateTregenzaMesh();
    InitRays((int)mRayDirections.size());
    CreateRays();

    info("Skydome instance created with default constructor, ready for raytracing.");
}

Skydome::Skydome(int skyType)
{
    mRayOrigin = {0.0, 0.0, 0.0};
    if (skyType == 1)
    {
        CreateTregenzaMesh();
    }
    else if (skyType == 2)
    {
        CreateReinhartMesh();
    }
    else
    {
        warning("Invalid sky type, defaulting to Tregenza mesh.");
        CreateTregenzaMesh();
    }

    InitRays((int)mRayDirections.size());
    CreateRays();
    info("Skydome instance created, ready for raytracing.");
}

Skydome::~Skydome()
{
    // Vectors clean themselves up automatically
}

void Skydome::InitRays(int rayCount)
{
    mRayCount = rayCount;

    debug("Skydome rays data:");
    debug("Number of rays:" + str(mRayCount) + ".");

    mRays.resize(mRayCount);
}

int Skydome::GetFaceCount()
{
    return (int)mFaces.size();
}

int Skydome::GetRayCount()
{
    return mRayCount;
}

std::vector<Ray> &Skydome::GetRays()
{
    return mRays;
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
    Vec3 origin(new_origin.x, new_origin.y, new_origin.z);
    for (int i = 0; i < mRayCount; i++)
    {
        mRays[i].org = origin;
    }
}

void Skydome::CreateRays()
{
    for (int i = 0; i < mRayCount; i++)
    {
        // BVH Ray constructor: Ray(origin, direction, tmin, tmax)
        Vec3 origin(mRayOrigin[0], mRayOrigin[1], mRayOrigin[2]);
        Vec3 direction(mRayDirections[i][0], mRayDirections[i][1], mRayDirections[i][2]);
        mRays[i] = Ray(origin, direction, 0.05f, std::numeric_limits<Scalar>::infinity());
    }
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
