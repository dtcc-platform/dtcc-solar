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
    BundleRays();

    info("Sunrays instance is setup and ready for raytracing.");
}

Sunrays::Sunrays(Vertex *faceMidPoints, int faceCount, std::vector<bool> faceMask)
{
    mRayCount = faceCount;
    mFaceMask = faceMask;
    InitRays(faceCount);
    CreateRays(faceMidPoints, faceCount);
    BundleRays();

    info("Sunrays instance is setup and ready for raytracing.");
}

Sunrays::~Sunrays()
{
    // Delete the 2d array
    for (int i = 0; i < mBundle8Count; i++)
        delete[] mRays8Valid[i];
    delete[] mRays8Valid;

    for (int i = 0; i < mBundle8Count; i++)
        delete[] mRays8ValidMask[i];
    delete[] mRays8ValidMask;
}

void Sunrays::InitRays(int rayCount)
{
    mRayCount = rayCount;
    mBundle8Count = ceil((float)mRayCount / 8.0f);

    debug("Sun rays data: \n");
    debug("Number of rays:" + str(mRayCount) + ".");
    debug("Number of 8 bundles:" + str(mBundle8Count) + ".");

    mRays1 = std::vector<RTCRay>(mRayCount);
    mRays8 = std::vector<RTCRay8>(mBundle8Count);

    // Defining a 2d array for the vadility of each ray in the 8 group bundles.
    mRays8Valid = new int *[mBundle8Count];
    mRays8ValidMask = new int *[mBundle8Count];
    for (int i = 0; i < mBundle8Count; i++)
    {
        mRays8Valid[i] = new int[8];
        mRays8ValidMask[i] = new int[8];
        for (int j = 0; j < 8; j++)
        {
            mRays8Valid[i][j] = 0;
            mRays8ValidMask[i][j] = 0;
        }
    }

    info("Sunrays initialized.");
}

int Sunrays::GetRayCount()
{
    return mRayCount;
}

int Sunrays::GetBundle8Count()
{
    return mBundle8Count;
}

std::vector<RTCRay> &Sunrays::GetRays()
{
    return mRays1;
}

std::vector<RTCRay8> &Sunrays::GetRays8()
{
    return mRays8;
}

int **Sunrays::GetValid8(bool applyMask)
{
    if (applyMask)
        return mRays8ValidMask;

    return mRays8Valid;
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

            mRays1[rayCounter].org_x = x;
            mRays1[rayCounter].org_y = y;
            mRays1[rayCounter].org_z = z;

            mRays1[rayCounter].dir_x = 0.0f;
            mRays1[rayCounter].dir_y = 0.0f;
            mRays1[rayCounter].dir_z = 1.0f;

            mRays1[rayCounter].tnear = 0;
            mRays1[rayCounter].tfar = std::numeric_limits<float>::infinity();
            mRays1[rayCounter].mask = -1;
            mRays1[rayCounter].flags = 0;

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
        mRays1[i].org_x = faceMidPts[i].x;
        mRays1[i].org_y = faceMidPts[i].y;
        mRays1[i].org_z = faceMidPts[i].z;

        mRays1[i].dir_x = 0.0f;
        mRays1[i].dir_y = 0.0f;
        mRays1[i].dir_z = 0.0f;

        mRays1[i].tnear = 0.05; // 5 cm
        mRays1[i].tfar = std::numeric_limits<float>::infinity();
        mRays1[i].mask = -1;
        mRays1[i].flags = 0;
    }
    info("Rays created from face mid points.");
}

void Sunrays::BundleRays()
{
    int bundleIndex8 = -1;
    int rayIndex8 = 0;

    /* Sort the rays in groups of 4, 8 and 16 */
    for (int i = 0; i < mRayCount; i++)
    {
        float x = mRays1[i].org_x;
        float y = mRays1[i].org_y;
        float z = mRays1[i].org_z;

        float dir_x = mRays1[i].dir_x;
        float dir_y = mRays1[i].dir_y;
        float dir_z = mRays1[i].dir_z;

        float tNear = mRays1[i].tnear;
        float tFar = mRays1[i].tfar;
        unsigned int mask = mRays1[i].mask;
        unsigned int flag = mRays1[i].flags;

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
        int faceIndex8 = bundleIndex8 * 8 + rayIndex8;
        if (mFaceMask[faceIndex8])
            mRays8ValidMask[bundleIndex8][rayIndex8] = -1;
    }
    info("Rays sorted in bundles of 8");
}

void Sunrays::UpdateRay1Directions(std::vector<float> new_sun_vec, bool applyMask)
{
    for (int i = 0; i < mRayCount; i++)
    {
        bool validRay = true;
        if (applyMask && !mFaceMask[i])
            validRay = false;

        if (validRay)
        {
            mRays1[i].dir_x = new_sun_vec[0];
            mRays1[i].dir_y = new_sun_vec[1];
            mRays1[i].dir_z = new_sun_vec[2];
        }
    }
}

void Sunrays::UpdateRay8Directions(std::vector<float> new_sun_vec, bool applyMask)
{
    for (int i = 0; i < mBundle8Count; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            int validRay = 0;
            if (applyMask)
                validRay = mRays8ValidMask[i][j];
            else
                validRay = mRays8Valid[i][j];

            if (validRay == -1)
            {
                mRays8[i].dir_x[j] = new_sun_vec[0];
                mRays8[i].dir_y[j] = new_sun_vec[1];
                mRays8[i].dir_z[j] = new_sun_vec[2];
            }
        }
    }
}
