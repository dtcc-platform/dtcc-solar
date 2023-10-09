import os
import trimesh
import numpy as np
import pandas as pd
import math
import copy

from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.sunpath_vis import SunpathMesh

from dtcc_solar.skydome import SkyDome

from pprint import pp
from dtcc_model import Mesh, PointCloud
from dtcc_viewer import Scene, Window, MeshShading

from dtcc_solar.utils import concatenate_meshes


class TestSkydome:
    lon: float
    lat: float
    solar_engine: SolarEngine
    skydome: SkyDome

    def setup_method(self):
        self.lon = -0.12
        self.lat = 22  # 51.5
        self.radius = 10
        self.file_name = "../data/models/CitySurfaceS.stl"
        self.w_file = "../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm"
        self.city_mesh = trimesh.load_mesh(self.file_name)
        self.solar_engine = SolarEngine(self.city_mesh)
        self.sunpath = Sunpath(self.lat, self.lon, self.radius)
        self.sunpath_mesh = SunpathMesh(self.radius)

    def test_skydome(self):
        skydome = SkyDome(self.radius, div_count=20)
        assert skydome

    def test_skydome_vis(self):
        self.skydome = SkyDome(self.radius, div_count=50)
        vs = self.skydome.dome_mesh.vertices
        fs = self.skydome.dome_mesh.faces
        dmesh = Mesh(vertices=vs, faces=fs)

        sun_pos_dict = self.sunpath.get_analemmas(2019, 2)
        pc = self.sunpath_mesh.create_sunpath_pc(sun_pos_dict)

        window = Window(1200, 800)
        scene = Scene()
        scene.add_mesh("Skydome", dmesh)
        scene.add_pointcloud("Analemmas", pc, size=0.08)
        window.render(scene)


if __name__ == "__main__":
    os.system("clear")
    print("--------------------- Raytracing test started -----------------------")

    test = TestSkydome()
    test.setup_method()
    # test.test_skydome()
    test.test_skydome_vis()
