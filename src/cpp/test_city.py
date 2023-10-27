import sys

sys.path.insert(0, "./build")
import py_embree_solar
import numpy as np
import time
from dtcc_io import meshes
from pprint import pp

mesh = meshes.load_mesh("../../data/models/CitySurfaceL.stl")

n_suns = 100

sun_vecs = np.zeros((n_suns, 3))

sun_vecs[:, 0] = np.linspace(0.5, 1.0, n_suns)
sun_vecs[:, 1] = np.linspace(0.5, 1.0, n_suns)
sun_vecs[:, 2] = np.linspace(0.5, 1.0, n_suns)

es = py_embree_solar.PyEmbreeSolar(mesh.vertices, mesh.faces, sun_vecs[0, :])

# run analysis
start_time = time.time()

results1 = es.iterateRaytrace_occ1(sun_vecs)
results2 = es.iterateRaytrace_occ4(sun_vecs)
results3 = es.iterateRaytrace_occ8(sun_vecs)
results4 = es.iterateRaytrace_occ16(sun_vecs)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time:.2f} seconds")
