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

    for (int i = 0; i < mBundle4Count; i++)
        delete[] mRays4ValidMask[i];
    delete[] mRays4ValidMask;

    for (int i = 0; i < mBundle8Count; i++)
        delete[] mRays8ValidMask[i];
    delete[] mRays8ValidMask;

    for (int i = 0; i < mBundle16Count; i++)
        delete[] mRays16ValidMask[i];
    delete[] mRays16ValidMask;
}

void Sunrays::InitRays(int rayCount)
{
    mRayCount = rayCount;
    mBundle4Count = ceil((float)mRayCount / 4.0f);
    mBundle8Count = ceil((float)mRayCount / 8.0f);
    mBundle16Count = ceil((float)mRayCount / 16.0f);

    debug("Sun rays data: \n");
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
    mRays4ValidMask = new int *[mBundle4Count];
    for (int i = 0; i < mBundle4Count; i++)
    {
        mRays4Valid[i] = new int[4];
        mRays4ValidMask[i] = new int[4];
        for (int j = 0; j < 4; j++)
        {
            mRays4Valid[i][j] = 0;
            mRays4ValidMask[i][j] = 0;
        }
    }

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

    // Defining a 2d array for the vadility of each ray in the 16 group bundles.
    mRays16Valid = new int *[mBundle16Count];
    mRays16ValidMask = new int *[mBundle16Count];
    for (int i = 0; i < mBundle16Count; i++)
    {
        mRays16Valid[i] = new int[16];
        mRays16ValidMask[i] = new int[16];
        for (int j = 0; j < 16; j++)
        {
            mRays16Valid[i][j] = 0;
            mRays16ValidMask[i][j] = 0;
        }
    }

    info("Sunrays initialized.");
}

int Sunrays::GetRayCount()
{
    return mRayCount;
}

int Sunrays::GetBundle4Count()
{
    return mBundle4Count;
}

int Sunrays::GetBundle8Count()
{
    return mBundle8Count;
}

int Sunrays::GetBundle16Count()
{
    return mBundle16Count;
}

std::vector<RTCRay> &Sunrays::GetRays()
{
    return mRays;
}

std::vector<RTCRay4> &Sunrays::GetRays4()
{
    return mRays4;
}

std::vector<RTCRay8> &Sunrays::GetRays8()
{
    return mRays8;
}

std::vector<RTCRay16> &Sunrays::GetRays16()
{
    return mRays16;
}

int **Sunrays::GetValid4(bool applyMask)
{
    if (applyMask)
        return mRays4ValidMask;

    return mRays4Valid;
}

int **Sunrays::GetValid8(bool applyMask)
{
    if (applyMask)
        return mRays8ValidMask;

    return mRays8Valid;
}

int **Sunrays::GetValid16(bool applyMask)
{
    if (applyMask)
        return mRays16ValidMask;

    return mRays16Valid;
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
    info("Rays created in a grid.");
}

void Sunrays::CreateRays(Vertex *faceMidPts, int faceCount)
{
    // Create rays from face mid pts and sun vector

    for (int i = 0; i < faceCount; i++)
    {
        mRays[i].org_x = faceMidPts[i].x;
        mRays[i].org_y = faceMidPts[i].y;
        mRays[i].org_z = faceMidPts[i].z;

        mRays[i].dir_x = 0.0f;
        mRays[i].dir_y = 0.0f;
        mRays[i].dir_z = 0.0f;

        mRays[i].tnear = 0.05; // 5 cm
        mRays[i].tfar = std::numeric_limits<float>::infinity();
        mRays[i].mask = -1;
        mRays[i].flags = 0;
    }
    info("Rays created from face mid points.");
}

void Sunrays::BundleRays()
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

        // Set the validity of the ray in the bundle, -1 = Valid, 0 = Invalid
        mRays4Valid[bundleIndex4][rayIndex4] = -1;
        int faceIndex4 = bundleIndex4 * 4 + rayIndex4;
        if (mFaceMask[faceIndex4])
            mRays4ValidMask[bundleIndex4][rayIndex4] = -1;

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
        int faceIndex16 = bundleIndex16 * 16 + rayIndex16;
        if (mFaceMask[faceIndex16])
            mRays16ValidMask[bundleIndex16][rayIndex16] = -1;
    }
    info("Rays sorted in bundles of 4, 8 and 16.");
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
            mRays[i].dir_x = new_sun_vec[0];
            mRays[i].dir_y = new_sun_vec[1];
            mRays[i].dir_z = new_sun_vec[2];
        }
    }
}

void Sunrays::UpdateRay4Directions(std::vector<float> new_sun_vec, bool applyMask)
{
    for (int i = 0; i < mBundle4Count; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            int validRay = 0;
            if (applyMask)
                validRay = mRays4ValidMask[i][j];
            else
                validRay = mRays4Valid[i][j];

            if (validRay == -1)
            {
                mRays4[i].dir_x[j] = new_sun_vec[0];
                mRays4[i].dir_y[j] = new_sun_vec[1];
                mRays4[i].dir_z[j] = new_sun_vec[2];
            }
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

void Sunrays::UpdateRay16Directions(std::vector<float> new_sun_vec, bool applyMask)
{
    for (int i = 0; i < mBundle16Count; i++)
    {
        for (int j = 0; j < 16; j++)
        {
            int validRay = 0;
            if (applyMask)
                validRay = mRays16ValidMask[i][j];
            else
                validRay = mRays16Valid[i][j];

            if (validRay == -1)
            {
                mRays16[i].dir_x[j] = new_sun_vec[0];
                mRays16[i].dir_y[j] = new_sun_vec[1];
                mRays16[i].dir_z[j] = new_sun_vec[2];
            }
        }
    }
}
