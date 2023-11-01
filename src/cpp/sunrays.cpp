#include "sunrays.h"

Sunrays::Sunrays()
{
    printf("Sunrays created with default constructor.\n");

    // Ray parameters
    mRp.xMin = -10.0f;
    mRp.xMax = 10.0f;
    mRp.yMin = -10.0f;
    mRp.yMax = 10.0f;
    mRp.xPadding = 0.1f;
    mRp.yPadding = 0.1f;
    mRp.xCount = 280;
    mRp.yCount = 280;

    mRayCount = mRp.xCount * mRp.yCount;

    InitRays(mRayCount);
    CreateGridRays();
    BundleRays();

    printf("Sunrays instance created, ready for raytracing.\n");
}

Sunrays::Sunrays(Vertex *faceMidPoints, int faceCount)
{
    mRayCount = faceCount;
    InitRays(faceCount);
    CreateRays(faceMidPoints, faceCount);
    BundleRays();

    printf("Sunrays instance created, ready for raytracing.\n");
}

Sunrays::~Sunrays()
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

void Sunrays::InitRays(int rayCount)
{
    mRayCount = rayCount;
    mBundle4Count = ceil((float)mRayCount / 4.0f);
    mBundle8Count = ceil((float)mRayCount / 8.0f);
    mBundle16Count = ceil((float)mRayCount / 16.0f);

    std::cout << "Sun rays data: " << std::endl;
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

RTCRay *Sunrays::GetRays()
{
    return mRays;
}

RTCRay4 *Sunrays::GetRays4()
{
    return mRays4;
}

RTCRay8 *Sunrays::GetRays8()
{
    return mRays8;
}

RTCRay16 *Sunrays::GetRays16()
{
    return mRays16;
}

int **Sunrays::GetValid4()
{
    return mRays4Valid;
}

int **Sunrays::GetValid8()
{
    return mRays8Valid;
}

int **Sunrays::GetValid16()
{
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

void Sunrays::UpdateRay1Directions(std::vector<float> new_sun_vec)
{
    for (int i = 0; i < mRayCount; i++)
    {
        mRays[i].dir_x = new_sun_vec[0];
        mRays[i].dir_y = new_sun_vec[1];
        mRays[i].dir_z = new_sun_vec[2];
    }
}

void Sunrays::UpdateRay4Directions(std::vector<float> new_sun_vec)
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

void Sunrays::UpdateRay8Directions(std::vector<float> new_sun_vec)
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

void Sunrays::UpdateRay16Directions(std::vector<float> new_sun_vec)
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
