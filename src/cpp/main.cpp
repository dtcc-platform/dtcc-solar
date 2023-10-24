
#include "embree_solar.h"

int main()
{
    EmbreeSolar *raytracer = new EmbreeSolar();

    // Setup analysis
    raytracer->createDevice();
    raytracer->createScene();
    raytracer->createGeomPlane();
    raytracer->createGridRays();
    raytracer->bundleRays();

    // Run analysis
    raytracer->raytrace_int1();
    raytracer->raytrace_occ1();
    raytracer->raytrace_occ4();
    raytracer->raytrace_occ8();
    raytracer->raytrace_occ16();

    return 0;
}