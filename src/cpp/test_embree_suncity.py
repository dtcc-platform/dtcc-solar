import sys

sys.path.insert(0, "./build")
import py_embree_solar
import numpy as np
import time
import math
from dtcc_io import meshes
from dtcc_model import Mesh, PointCloud
from pprint import pp
from dtcc_viewer import Scene, Window, MeshShading

mesh = meshes.load_mesh("../../data/models/CitySurfaceL.stl")

n_suns = 1000

r = 500
rot_step = np.linspace(0, 1 * np.pi, n_suns)
x = r * np.sin(rot_step)
y = r * np.cos(rot_step)
z = r * abs(np.sin(rot_step / 1.0))
sun_vecs = np.c_[x, y, z]

es = py_embree_solar.PyEmbreeSolar(mesh.vertices, mesh.faces)

faces = es.getSkydomeFaces()
vertices = es.getSkydomeVertices()
skydome_rays = es.getSkydomeRays()

# run analysis

results1 = es.sun_raytrace_occ1(sun_vecs)
# results2 = es.sun_raytrace_occ4(sun_vecs)
# results3 = es.sun_raytrace_occ8(sun_vecs)
# results4 = es.sun_raytrace_occ16(sun_vecs)

results = np.sum(-1 * results1, axis=0)

pc = PointCloud(points=sun_vecs)

print("Face count:" + str(len(mesh.faces)))
print("Results count:" + str(len(results)))

window = Window(1200, 800)
scene = Scene()
scene.add_pointcloud("pc", pc, size=1)
scene.add_mesh("Skydome", mesh, results)
window.render(scene)
