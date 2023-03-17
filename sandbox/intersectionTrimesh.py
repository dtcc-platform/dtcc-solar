import numpy as np
import math
import trimesh
import matplotlib.pyplot as plt
import utils
import time


def PreProcess(mesh, sunVecRev, rayOrigins, rayDirections, faceMidPoints):
    #Get all ray origins
    for i in range(0, len(mesh.faces)):
        face = mesh.faces[i]
        v1 = mesh.vertices[face[0]]
        v2 = mesh.vertices[face[1]]
        v3 = mesh.vertices[face[2]]
        vAvrg = utils.AvrgVertex(v1,v2,v3)
        rayOrigins.append(vAvrg)
        rayDirections.append(sunVecRev)
        faceMidPoints.append(vAvrg)


def RayTrace(mesh, sunVec, sunVecRev):

    rayOrigins = []
    rayDirections = []
    faceMidPoints = []

    PreProcess(mesh, sunVecRev, rayOrigins, rayDirections, faceMidPoints)

    time2 = time.perf_counter()
    [locations, index_ray, index_tri] = mesh.ray.intersects_location(ray_origins = rayOrigins, ray_directions = rayDirections)
    time3 = time.perf_counter()
    print("Mesh ray intersection time: "  + str(round(time3  - time2,2)))

    faceRayFaces = {}

    PostProcess(faceRayFaces, index_ray, index_tri, )

    return faceRayFaces, faceMidPoints


def PostProcess(faceRayFaces, index_ray, index_tri):

    #Rearrange intersection results
    for i in range(0, len(index_ray)):
        rayIndex = index_ray[i]                         #Ray index = face index, since there is one ray per face
        indexOfIntersectingFace = index_tri[i]
        if rayIndex != indexOfIntersectingFace:
            if rayIndex in faceRayFaces:
                faceRayFaces[rayIndex].append(indexOfIntersectingFace)
            else:
                faceRayFaces[rayIndex] = []
                faceRayFaces[rayIndex].append(indexOfIntersectingFace)    

    
