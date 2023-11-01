#pragma once
#include "embree_solar.h"

int main()
{
    // Create raytracer instance
    EmbreeSolar *raytracer = new EmbreeSolar();

    // Sun vector 1
    std::vector<float> sunVec1 = {0.0, 0.0, 1.0};
    std::vector<float> sunVec2 = {0.0, 1.0, 1.0};
    std::vector<float> sunVec3 = {1.0, 1.0, 1.0};

    // All sun vectors
    std::vector<std::vector<float>> sun_vecs;
    sun_vecs.push_back(sunVec1);
    sun_vecs.push_back(sunVec2);
    sun_vecs.push_back(sunVec3);

    // Run analysis
    raytracer->SunRaytrace_Occ1(sun_vecs);
    raytracer->SunRaytrace_Occ4(sun_vecs);
    raytracer->SunRaytrace_Occ8(sun_vecs);
    raytracer->SunRaytrace_Occ16(sun_vecs);

    return 0;
}