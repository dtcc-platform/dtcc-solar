#include "sunrays.h"

Sunrays::Sunrays()
{
    printf("Sunrays created with default constructor.");

    // Ray parameters
    mRp.xMin = -10.0f;
    mRp.xMax = 10.0f;
    mRp.yMin = -10.0f;
    mRp.yMax = 10.0f;
    mRp.xPadding = 0.1f;
    mRp.yPadding = 0.1f;
    mRp.xCount = 201;
    mRp.yCount = 201;

    mRayCount = mRp.xCount * mRp.yCount;
    mFaceMask = std::vector<bool>(mRayCount, true);

    InitRays(mRayCount);
    CreateGridRays();

    info("Sunrays instance is setup and ready for raytracing.");
}

Sunrays::Sunrays(Vertex *faceMidPoints, int faceCount, std::vector<bool> faceMask)
{
    mRayCount = faceCount;
    mFaceMask = faceMask;
    InitRays(faceCount);
    CreateRays(faceMidPoints, faceCount);

    info("Sunrays instance is setup and ready for raytracing.");
}

Sunrays::~Sunrays()
{
    // Vectors clean themselves up automatically
}

void Sunrays::InitRays(int rayCount)
{
    mRayCount = rayCount;

    debug("Sun rays data:");
    debug("Number of rays:" + str(mRayCount) + ".");

    mRays.resize(mRayCount);

    info("Sunrays initialized.");
}

int Sunrays::GetRayCount()
{
    return mRayCount;
}

std::vector<Ray> &Sunrays::GetRays()
{
    return mRays;
}

void Sunrays::CreateGridRays()
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

            // BVH Ray constructor: Ray(origin, direction, tmin, tmax)
            Vec3 origin(x, y, z);
            Vec3 direction(0.0f, 0.0f, 1.0f);
            mRays[rayCounter] = Ray(origin, direction, 0.0f, std::numeric_limits<Scalar>::infinity());

            rayCounter++;
        }
    }
    info("Rays created in a grid.");
}

void Sunrays::CreateRays(Vertex *faceMidPts, int faceCount)
{
    // Create rays from face mid pts and sun vector

    for (int i = 0; i < faceCount; i++)
    {
        // BVH Ray constructor: Ray(origin, direction, tmin, tmax)
        Vec3 origin(faceMidPts[i].x, faceMidPts[i].y, faceMidPts[i].z);
        Vec3 direction(0.0f, 0.0f, 0.0f); // Direction set later via UpdateRayDirections
        mRays[i] = Ray(origin, direction, 0.05f, std::numeric_limits<Scalar>::infinity());
    }
    info("Rays created from face mid points.");
}

void Sunrays::UpdateRayDirections(std::vector<float> new_sun_vec, bool applyMask)
{
    Vec3 direction(new_sun_vec[0], new_sun_vec[1], new_sun_vec[2]);
    for (int i = 0; i < mRayCount; i++)
    {
        bool validRay = true;
        if (applyMask && !mFaceMask[i])
            validRay = false;

        if (validRay)
        {
            mRays[i].dir = direction;
        }
    }
}
