import numpy as np
import trimesh
import os

os.system("clear")

path = "data/models/"
fileNames = [
    "City.stl",
    "City66k.stl",
    "CitySurface.vtu",
    "CitySurface626k.stl",
    "City136kSoft.stl",
    "CitySurface_S.stl",
    "CitySurface_XS.stl",
    "CitySurface69k.stl",
]
fileName = path + fileNames[7]
mesh = trimesh.load_mesh(fileName)

pt1 = [0, 0, 0]
pt2 = [10, 0, 0]
pt3 = [10, 10, 0]

sphere1 = trimesh.primitives.Sphere(radius=1, center=pt1)
sphere2 = trimesh.primitives.Sphere(radius=1, center=pt2)
sphere1.visual.face_colors = [1.0, 1.0, 0, 0.8]
sphere2.visual.face_colors = [1.0, 1.0, 0, 0.8]

scene = trimesh.Scene()
scene.add_geometry(sphere1)
scene.add_geometry(sphere2)

vs = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [10, 10, 10]], dtype=float)

cs = np.array([[1.0, 1.0, 0, 0.8], [1.0, 1.0, 0, 0.8], [1.0, 1.0, 0, 0.8]], dtype=float)

cs1 = np.array([[0.0, 1.0, 1.0, 0.8]], dtype=float)

cs2 = np.array([[1.0, 1.0, 0, 0.8]], dtype=float)

es = np.zeros((2, 2), dtype=int)
es[0, 0] = 0
es[1, 0] = 1
es[0, 1] = 2

el1 = trimesh.path.entities.Line([0, 1, 2, 0], color=cs1)

el2 = trimesh.path.entities.Line([0, 3, 2, 3], color=cs2)


print(vs)

path = trimesh.path.Path3D(entities=[el1, el2], vertices=vs)
scene.add_geometry(path)
scene.show()
