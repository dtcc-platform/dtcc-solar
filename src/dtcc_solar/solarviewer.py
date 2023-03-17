import sys
import pathlib
project_dir = str(pathlib.Path(__file__).resolve().parents[0])
sys.path.append(project_dir)

import numpy as np
import trimesh
import utils
import mesh_compute as mc

class SolarViewer:

    def __init__(self):
        self.scene = trimesh.Scene()
        self.scene.camera._fov = [35,35]     #Field of view [x, y]
        self.scene.camera.z_far = 10000      #Distance to the far clipping plane        

    def add_mesh(self, meshes):
        self.scene.add_geometry(meshes)

    def add_meshes(self, city_mesh, sun_mesh, sun_path_meshes):
        self.scene.add_geometry(city_mesh)
        self.scene.add_geometry(sun_mesh)
        self.scene.add_geometry(sun_path_meshes)

    def add_dome_mesh(self, dome_mesh):
        self.scene.add_geometry(dome_mesh)            

    def show(self):
        self.scene.show()               

    def create_solar_sphere(self, sunPos, sunSize):
        sunMesh = trimesh.primitives.Sphere(radius = sunSize,  center = sunPos, subdivisions = 4)
        sunMesh.visual.face_colors = [1.0, 0.5, 0, 1.0]
        return sunMesh            

    def create_solar_spheres(self, sunPos, sunSize):
        sunMeshes = []
        for i in range(0,len(sunPos)):
            sunMesh = trimesh.primitives.Sphere(radius = sunSize, center = sunPos[i,:], subdivisions = 1)
            sunMesh.visual.face_colors = [1.0, 0.5, 0, 1.0]
            sunMeshes.append(sunMesh)
        return sunMeshes

    def create_sunpath_loops(self, x, y, z, radius):
        path_meshes = []
        for h in x:
            vs = np.zeros((len(x[h])+1, 3))
            vi = np.zeros((len(x[h])),dtype=int)
            lines = []
            colors = []
            
            for i in range(0, len(x[h])):
                sunPos = [x[h][i], y[h][i], z[h][i]]
                vs[i,:] = sunPos
                vi[i] = i
                index2 = i + 1
                color = utils.GetBlendedSunColor(radius, z[h][i])
                colors.append(color)
                line = trimesh.path.entities.Line([i, index2])
                lines.append(line)

            vs[len(x[h]),:] = vs[0,:]     

            path = trimesh.path.Path3D(entities=lines, vertices=vs, colors= colors)
            path_meshes.append(path)

        return path_meshes        

