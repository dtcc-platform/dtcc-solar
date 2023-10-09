import numpy as np
import pandas as pd
import math
from dtcc_model import Mesh, PointCloud
from dtcc_solar.utils import calc_rotation_matrix
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.sunpath_vis import SunpathVis


class SkyCylinder:
    radius: float
    div_tangent: int
    div_height: int
    center: np.ndarray

    def __init__(self, radius: float, div_tangent: int = 10, div_height: int = 10):
        self.radius = radius
        self.div_tangent = div_tangent
        self.div_height = div_height
        self.center = np.array([0, 0, 0])
        self.lon = -0.12
        self.lat = 51.5
        self.sunpath = Sunpath(self.lat, self.lon, self.radius)

    def create_skycylinder_mesh(self):
        dh = self.radius / self.div_height
        da = (math.pi * 2) / self.div_tangent
        points = []
        faces = []

        index = 0
        r_count = self.div_tangent

        for i in range(self.div_height):
            h = self.center[2] + (i * dh)
            for j in range(self.div_tangent):
                angle = j * da
                x = self.center[0] + self.radius * math.cos(angle)
                y = self.center[1] + self.radius * math.sin(angle)
                z = h
                points.append(np.array([x, y, z]))
                if i > 0:
                    if j > 0:
                        f1 = r_count * i + j
                        f2 = r_count * i + (j - 1)
                        f3 = r_count * (i - 1) + j
                        faces.append([f1, f2, f3])

                    if j == (self.div_tangent - 1):
                        f1 = r_count * i + j - (r_count - 1)
                        f2 = r_count * i + j
                        f3 = r_count * (i - 1) + j - (r_count - 1)
                        faces.append([f1, f2, f3])

                    if j > 0:
                        f1 = r_count * (i - 1) + (j - 1)
                        f2 = r_count * i + (j - 1)
                        f3 = r_count * (i - 1) + j
                        faces.append([f1, f3, f2])

                    if j == (self.div_tangent - 1):
                        f1 = r_count * (i - 1) + (j)
                        f2 = r_count * i + (j)
                        f3 = r_count * (i - 1) + j - (r_count - 1)
                        faces.append([f1, f3, f2])

        points = np.array(points)
        faces = np.array(faces)
        self.mesh = Mesh(vertices=points, faces=faces)
        self.pc = PointCloud(points=points)

    def create_skycylinder_from_day_paths(self):
        dates = pd.date_range(start="2019-01-01", end="2019-12-31", freq="1D")
        sun_pos_dict = self.sunpath.get_daypaths(dates, 1)
        points = []
        for day in sun_pos_dict:
            for sun in sun_pos_dict[day]:
                print(type(sun))
                points.append(np.array([sun.x, sun.y, sun.z]))

        points = np.array(points)

        return points

    def tilt(self, vec_from: np.ndarray, vec_to: np.ndarray):
        R = calc_rotation_matrix(vec_from, vec_to)
        vertices = self.mesh.vertices
        new_vertices = []
        for i in range(len(vertices)):
            new_vertex = np.matmul(R, vertices[i, :])
            new_vertices.append(new_vertex)

        new_vertices = np.array(new_vertices)
        self.mesh.vertices = new_vertices

    def project_to_sphere(self):
        vertices = self.mesh.vertices
        new_vertices = []
        for i in range(len(vertices)):
            vertex = vertices[i, :]
