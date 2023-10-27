
#include </opt/homebrew/Cellar/embree/4.3.0/include/embree4/rtcore.h>
#include <stdio.h>
#include <math.h>
#include <limits>
#include <iostream>
#include <vector>
#include "common.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class EmbreeSolar
{

public:
    EmbreeSolar();
    EmbreeSolar(std::vector<std::vector<float>> vertices, std::vector<std::vector<int>> faces, std::vector<float> sun_vec);
    ~EmbreeSolar();

    void initArrays(int rayCount);
    void release();
    void createDevice();
    void createScene();
    void createGeom(std::vector<std::vector<float>> vertices, std::vector<std::vector<int>> faces);
    void createGeomPlane();
    void createGridRays();
    void createRaysFromFaces(std::vector<float> sun_vec);
    void bundleRays();

    void updateRay1Directions(std::vector<float> new_sun_vec);
    void updateRay4Directions(std::vector<float> new_sun_vec);
    void updateRay8Directions(std::vector<float> new_sun_vec);
    void updateRay16Directions(std::vector<float> new_sun_vec);

    void raytrace_occ1();
    void raytrace_occ4();
    void raytrace_occ8();
    void raytrace_occ16();

    void raytrace_occ1_2(std::vector<int> &results, int &hitCounter);
    void raytrace_occ4_2(std::vector<int> &results, int &hitCounter);
    void raytrace_occ8_2(std::vector<int> &results, int &hitCounter);
    void raytrace_occ16_2(std::vector<int> &results, int &hitCounter);

    std::vector<std::vector<int>> iterateRaytrace_occ1(std::vector<std::vector<float>> sun_vecs);
    std::vector<std::vector<int>> iterateRaytrace_occ4(std::vector<std::vector<float>> sun_vecs);
    std::vector<std::vector<int>> iterateRaytrace_occ8(std::vector<std::vector<float>> sun_vecs);
    std::vector<std::vector<int>> iterateRaytrace_occ16(std::vector<std::vector<float>> sun_vecs);

    void errorFunction(void *userPtr, enum RTCError error, const char *str);

private:
    RTCScene mScene;
    RTCDevice mDevice;
    RTCGeometry mGeometry;

    Parameters mPp; // plane parameters
    Parameters mSp; // sphere parameters
    Parameters mRp; // ray parameters

    int mRayCount;
    int mBundle4Count;
    int mBundle8Count;
    int mBundle16Count;

    int mVertexCount;
    int mFaceCount;

    Face *mFaces;
    Vertex *mVertices;
    Vertex *mFaceMidPts;

    RTCRay *mRays;
    RTCRay4 *mRays4;
    RTCRay8 *mRays8;
    RTCRay16 *mRays16;

    int **mRays4Valid;
    int **mRays8Valid;
    int **mRays16Valid;
};
