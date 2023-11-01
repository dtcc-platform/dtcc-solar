import sys

sys.path.insert(0, "./build")
import py_embree_solar
import numpy as np
import time
from dtcc_io import meshes
from dtcc_model import Mesh, PointCloud
from pprint import pp
from dtcc_viewer import Scene, Window, MeshShading

mesh = meshes.load_mesh("../../data/models/CitySurfaceL.stl")

n_suns = 1000

sun_vecs = np.zeros((n_suns, 3))
sun_vecs[:, 0] = np.linspace(0.5, 1.0, n_suns)
sun_vecs[:, 1] = np.linspace(0.5, 1.0, n_suns)
sun_vecs[:, 2] = np.linspace(0.5, 1.0, n_suns)

es = py_embree_solar.PyEmbreeSolar(mesh.vertices, mesh.faces)

faces = es.getSkydomeFaces()
vertices = es.getSkydomeVertices()
skydome_rays = es.getSkydomeRays()

pc = PointCloud(points=skydome_rays)

mesh = Mesh(vertices=vertices, faces=faces)

# window = Window(1200, 800)
# scene = Scene()
# scene.add_pointcloud("pc", pc, size=0.01)
# scene.add_mesh("Skydome", mesh)
# window.render(scene)

# run analysis
start_time = time.time()

results1 = es.sun_raytrace_occ1(sun_vecs)
results2 = es.sun_raytrace_occ4(sun_vecs)
results3 = es.sun_raytrace_occ8(sun_vecs)
results4 = es.sun_raytrace_occ16(sun_vecs)

end_time = time.time()
elapsed_time = end_time - start_time

# print(f"Elapsed time: {elapsed_time:.2f} seconds")
