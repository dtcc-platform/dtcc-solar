#pragma once
#include </opt/homebrew/Cellar/embree/4.3.0/include/embree4/rtcore.h>
#include <stdio.h>
#include <math.h>
#include <limits>
#include <iostream>
#include <vector>
#include "common.h"

class EmbreeSolar
{

public:
    EmbreeSolar();
    ~EmbreeSolar();

    void initArrays();
    void release();
    void createDevice();
    void createScene();
    void createGeom(Vertex *vertices, int nVertices, Face *triangles, int nFaces);
    void createGeomPlane();
    void createGridRays();
    void bundleRays();
    void raytrace_int1();
    void raytrace_occ1();
    void raytrace_occ4();
    void raytrace_occ8();
    void raytrace_occ16();

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

    Vertex *mVertices;
    Face *mFaces;

    RTCRayHit *mRays;

    RTCRay4 *mRays4;
    RTCRay8 *mRays8;
    RTCRay16 *mRays16;

    int **mRays4Valid;
    int **mRays8Valid;
    int **mRays16Valid;
};
