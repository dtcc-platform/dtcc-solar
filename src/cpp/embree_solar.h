#pragma once
#include </opt/homebrew/Cellar/embree/4.3.0/include/embree4/rtcore.h>
#include <stdio.h>
#include <math.h>
#include <limits>
#include <iostream>
#include <vector>
#include "common.h"
#include "sunrays.h"
#include "skydome.h"

#ifdef PYTHON_MODULE
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

class EmbreeSolar
{

public:
    EmbreeSolar();
    EmbreeSolar(std::vector<std::vector<float>> vertices, std::vector<std::vector<int>> faces);
    ~EmbreeSolar();

    void initRays(int rayCount);
    void CreateDevice();
    void CreateScene();
    void CreateGeom(std::vector<std::vector<float>> vertices, std::vector<std::vector<int>> faces);
    void CreateGeomPlane();
    void createGridRays();
    void createRaysFromFaces();
    void bundleRays();
    void CalcFaceMidPoints();

    std::vector<std::vector<int>> GetSkydomeFaces();
    std::vector<std::vector<float>> GetSkydomeVertices();
    std::vector<std::vector<float>> GetSkydomeRayDirections();

    void updateRay1Directions(std::vector<float> new_sun_vec);
    void updateRay4Directions(std::vector<float> new_sun_vec);
    void updateRay8Directions(std::vector<float> new_sun_vec);
    void updateRay16Directions(std::vector<float> new_sun_vec);

    void Raytrace_occ1(std::vector<int> &results, int &hitCounter);
    void Raytrace_occ4(std::vector<int> &results, int &hitCounter);
    void Raytrace_occ8(std::vector<int> &results, int &hitCounter);
    void Raytrace_occ16(std::vector<int> &results, int &hitCounter);

    std::vector<std::vector<int>> SunRaytrace_Occ1(std::vector<std::vector<float>> sun_vecs);
    std::vector<std::vector<int>> SunRaytrace_Occ4(std::vector<std::vector<float>> sun_vecs);
    std::vector<std::vector<int>> SunRaytrace_Occ8(std::vector<std::vector<float>> sun_vecs);
    std::vector<std::vector<int>> SunRaytrace_Occ16(std::vector<std::vector<float>> sun_vecs);

    std::vector<float> SkyRaytrace_Occ1();
    std::vector<float> SkyRaytrace_Occ4();
    std::vector<float> SkyRaytrace_Occ8();
    std::vector<float> SkyRaytrace_Occ16();

    void ErrorFunction(void *userPtr, enum RTCError error, const char *str);

private:
    Skydome *mSkydome = NULL;
    Sunrays *mSunrays = NULL;

    RTCScene mScene;
    RTCDevice mDevice;
    RTCGeometry mGeometry;

    Parameters mPp; // plane parameters
    Parameters mSp; // sphere parameters
    Parameters mRp; // ray parameters

    int mVertexCount;
    int mFaceCount;

    Face *mFaces;
    Vertex *mVertices;
    Vertex *mFaceMidPts;
};
