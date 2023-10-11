import os
import trimesh
import numpy as np
import pandas as pd
import math

from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.sunpath_vis import SunpathMesh
from dtcc_solar.skycylinder import SkyCylinder

from pprint import pp
from dtcc_model import Mesh, PointCloud
from dtcc_viewer import Scene, Window, MeshShading
from dtcc_solar.utils import calc_rotation_matrix


class TestSkycylinder:
    lon: float
    lat: float
    solar_engine: SolarEngine
    skycylinder: SkyCylinder

    def setup_method(self):
        self.lon = -0.12
        self.lat = 51.5
        self.radius = 10
        self.file_name = "../data/models/CitySurfaceS.stl"
        self.w_file = "../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm"
        self.city_mesh = trimesh.load_mesh(self.file_name)
        self.solar_engine = SolarEngine(self.city_mesh)
        self.sunpath = Sunpath(self.lat, self.lon, self.radius)
        self.sunpath_mesh = SunpathMesh(self.radius)

    def test_skycylinder_day_loops(self):
        self.skycylinder = SkyCylinder(
            self.sunpath,
            self.solar_engine.horizon_z,
            150,
            20,
        )

        mesh = self.skycylinder.mesh
        pc_quad_mid_pts = self.skycylinder.pc

        sun_pos_dict = self.sunpath.get_analemmas(2019, 2)
        pc_analemmas = self.sunpath_mesh.create_sunpath_pc(sun_pos_dict)

        window = Window(1200, 800)
        scene = Scene()
        scene.add_pointcloud("Points", pc_quad_mid_pts, size=0.02)
        scene.add_pointcloud("Analemmas", pc_analemmas, size=0.08)
        scene.add_mesh("Mesh", mesh, shading=MeshShading.wireframe)
        window.render(scene)


if __name__ == "__main__":
    os.system("clear")
    print("--------------------- Skycylinder test started -----------------------")

    test = TestSkycylinder()
    test.setup_method()
    test.test_skycylinder_day_loops()
