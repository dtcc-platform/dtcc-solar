import numpy as np
import trimesh
import dtcc_solar.mesh_compute as mc
from dtcc_solar import utils

class Viewer:

    def __init__(self):
        self.scene = trimesh.Scene()
        self.scene.camera._fov = [35,35]     #Field of view [x, y]
        self.scene.camera.z_far = 10000      #Distance to the far clipping plane        

    def add_meshes(self, meshes):
        self.scene.add_geometry(meshes)

    def add_dome_mesh(self, dome_mesh):
        self.scene.add_geometry(dome_mesh)            

    def show(self):
        self.scene.show()               


