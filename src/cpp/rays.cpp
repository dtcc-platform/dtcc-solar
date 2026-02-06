#include "rays.h"

Rays::Rays(fArray2D rays)
{
    mRayOrigin = {0.0, 0.0, 0.0};
    mRayCount = (int)rays.size();

    InitRays(rays);
    CreateRays();
    info("Rays instance created, ready for raytracing.");
}

Rays::Rays(fArray2D rays, fArray1D solidAngles)
{
    mRayOrigin = {0.0, 0.0, 0.0};
    mRayCount = (int)rays.size();
    mRaySolidAngles = solidAngles;

    InitRays(rays);
    CreateRays();
    info("Rays instance created, ready for raytracing.");
}

Rays::~Rays()
{
    // Vectors clean themselves up automatically
    mRayDirections.clear();
    mRays.clear();
    mRaySolidAngles.clear();
}

const fArray1D &Rays::GetSolidAngles() const
{
    return mRaySolidAngles;
}

const std::vector<Ray> &Rays::GetRays() const
{
    return mRays;
}

const int Rays::GetRayCount() const
{
    return mRayCount;
}

const fArray2D &Rays::GetRayDirections() const
{
    return mRayDirections;
}

void Rays::InitRays(fArray2D rays)
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
            error("Invalid vertex size in Rays::InitRays.");
    }

    debug("Rays data:");
    debug("Number of rays:" + str(mRayCount) + ".");

    mRays.resize(mRayCount);
}

void Rays::CreateRays()
{
    for (int i = 0; i < mRayCount; i++)
    {
        // BVH Ray constructor: Ray(origin, direction, tmin, tmax)
        Vec3 origin(mRayOrigin[0], mRayOrigin[1], mRayOrigin[2]);
        Vec3 direction(mRayDirections[i][0], mRayDirections[i][1], mRayDirections[i][2]);
        mRays[i] = Ray(origin, direction, 0.05f, std::numeric_limits<Scalar>::infinity());
    }
}
