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

    return 0;
}