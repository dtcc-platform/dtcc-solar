#include "pydome.h"

Pydome::Pydome(fArray2D rays)
{
    mRayOrigin = {0.0, 0.0, 0.0};
    mRayCount = (int)rays.size();

    InitRays(rays);
    CreateRays();
    BundleRays();
    info("Pydome instance created, ready for raytracing.");
}

Pydome::Pydome(fArray2D rays, fArray1D solidAngles)
{
    mRayOrigin = {0.0, 0.0, 0.0};
    mRayCount = (int)rays.size();
    mRaySolidAngle = solidAngles;

    InitRays(rays);
    CreateRays();
    BundleRays();
    info("Pydome instance created, ready for raytracing.");
}

std::vector<float> Pydome::GetSolidAngles()
{
    return mRaySolidAngle;
}
std::vector<RTCRay> &Pydome::GetRays()
{
    return mRays;
}

std::vector<RTCRay8> &Pydome::GetRays8()
{
    return mRays8;
}

int **Pydome::GetValid8()
{
    return mRays8Valid;
}

int Pydome::GetRayCount()
{
    return mRayCount;
}

int Pydome::GetBundle8Count()
{
    return mBundle8Count;
}

fArray2D Pydome::GetRayDirections()
{
    return mRayDirections;
}

float Pydome::GetDomeSolidAngle()
{
    return mDomeSolidAngle;
}

void Pydome::InitRays(fArray2D rays)
{
    for (long unsigned int i = 0; i < rays.size(); i++)
    {
        if (rays[i].size() == 3)
        {
            std::vector<float> rayDir = {rays[i][0], rays[i][1], rays[i][2]};
            rayDir = UnitizeVector(rayDir); // Ensure the ray direction is a unit vector
            mRayDirections.push_back(rayDir);
        }
        else
            error("Invalid vertex size in constructor EmbreeSolar::Pydome.");
    }

    mBundle8Count = ceil((float)mRayCount / 8.0f);

    debug("Skydome rays data:");
    debug("Number of rays:" + str(mRayCount) + ".");
    debug("Number of 8 bundles:" + str(mBundle8Count) + ".");

    mRays = std::vector<RTCRay>(mRayCount);
    mRays8 = std::vector<RTCRay8>(mBundle8Count);

    // Defining a 2d array for the vadility of each ray in the 8 group bundles.
    mRays8Valid = new int *[mBundle8Count];
    for (int i = 0; i < mBundle8Count; i++)
    {
        mRays8Valid[i] = new int[8];
        for (int j = 0; j < 8; j++)
            mRays8Valid[i][j] = 0;
    }
}

void Pydome::CreateRays()
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

void Pydome::BundleRays()
{
    int bundleIndex8 = -1;
    int rayIndex8 = 0;

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
    }
}

void Pydome::TranslateRays(Vertex new_origin)
{
    for (int i = 0; i < mRayCount; i++)
    {
        mRays[i].org_x = new_origin.x;
        mRays[i].org_y = new_origin.y;
        mRays[i].org_z = new_origin.z;
    }
}

void Pydome::Translate8Rays(Vertex new_origin)
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