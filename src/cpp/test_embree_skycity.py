import sys

sys.path.insert(0, "./build")
import py_embree_solar
import numpy as np
import time
from dtcc_io import meshes
from pprint import pp
from dtcc_solar.skydome import SkyDome
from dtcc_model import Mesh, PointCloud
from dtcc_viewer import Scene, Window, MeshShading

mesh = meshes.load_mesh("../../data/models/CitySurfaceL.stl")
# mesh = meshes.load_mesh("../../data/models/City136kSoft.stl")

es = py_embree_solar.PyEmbreeSolar(mesh.vertices, mesh.faces)

faces = es.getSkydomeFaces()
vertices = es.getSkydomeVertices()
skydome_rays = es.getSkydomeRays()

pc = PointCloud(points=skydome_rays)
mesh_dome = Mesh(vertices=vertices, faces=faces)

results1 = es.sky_raytrace_occ1()
# results2 = es.sky_raytrace_occ4()
# results3 = es.sky_raytrace_occ8()
# results4 = es.sky_raytrace_occ16()

results = 1.0 - results1

window = Window(1200, 800)
scene = Scene()
# scene.add_pointcloud("pc", pc, size=0.01)
scene.add_mesh("City mesh", mesh, data=results)
window.render(scene)
