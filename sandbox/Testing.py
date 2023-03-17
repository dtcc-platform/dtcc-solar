
import numpy as np
import math
import trimesh
import os
import matplotlib.pyplot as plt
import pyglet
import utils
import time
import intersectNcollpyde as intNcoll
from ncollpyde import Volume
import meshio

os.system('clear')

print("-------- Solar Radiation Analysis Started -------")
time0 = time.perf_counter()
fileName = 'City.stl'
mesh = trimesh.load_mesh(fileName)
scene = trimesh.Scene()

origin = [0, 0, 0]
sunPos = [-10, -10, 5]
sunVec = utils.VectorFromPoints(sunPos, origin)
sunVec = utils.NormaliseVector(sunVec)
sunVecRev = utils.reverse_vector(sunVec)
sunMesh = trimesh.primitives.Sphere(radius = 1, center = sunPos)

faceColors = []
vertexColors = []

faceAngles = np.array([])
vertexAngles = np.array([])
maxFaceAngle = -100000
minFaceAngle = 100000

print("Mesh face count: " + str(len(mesh.faces)))
print("Mesh vertex count: " + str(len(mesh.vertices)))
print("Sun vector: " + str(sunVec))

time1 = time.perf_counter()
print("Mesh loading time: "  + str(round(time1  - time0,2)))

vCount = len(mesh.vertices)
fCount = len(mesh.faces)

#mesh = meshio.read(fileName)
#volume = Volume(mesh.points, mesh.cells_dict["triangle"])

intNcoll.RayTrace(fileName, sunVec, sunVecRev)


