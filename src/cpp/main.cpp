
#include "embree_solar.h"

int main()
{
    // Create raytracer instance
    EmbreeSolar *raytracer = new EmbreeSolar();

    // Run analysis
    raytracer->raytrace_int1();
    raytracer->raytrace_occ1();
    raytracer->raytrace_occ4();
    raytracer->raytrace_occ8();
    raytracer->raytrace_occ16();

    return 0;
}