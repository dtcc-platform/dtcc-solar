import os
import trimesh
import numpy as np
import pandas as pd
import math
import copy

from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.sunpath_vis import SunpathMesh

from dtcc_solar.skysphere import SkySphere

from pprint import pp
from dtcc_model import Mesh, PointCloud
from dtcc_io import meshes
from dtcc_viewer import Scene, Window, MeshShading
from dtcc_solar.utils import Parameters

from dtcc_solar.utils import concatenate_meshes


class TestSkysphere:
    lon: float
    lat: float
    solar_engine: SolarEngine

    def setup_method(self):
        self.lon = -0.12
        self.lat = 52
        self.radius = 10
        self.file_name = "../data/models/CitySurfaceS.stl"
        self.w_file = "../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm"
        self.city_mesh = trimesh.load_mesh(self.file_name)
        self.solar_engine = SolarEngine(self.city_mesh)
        self.sunpath = Sunpath(self.lat, self.lon, self.radius)
        self.sunpath_mesh = SunpathMesh(self.radius)
        self.p = Parameters(
            file_name=self.file_name,
            latitude=self.lat,
            longitude=self.lon,
            weather_file=self.w_file,
        )

    def test_skysphere(self):
        skysphere = SkySphere(self.radius, div_count=20)
        assert skysphere

    def test_skysphere_vis(self):
        sun_pos_dict = self.sunpath.get_analemmas(2019, 2)
        pc = self.sunpath_mesh.create_sunpath_pc(sun_pos_dict)

        self.skysphere = SkySphere(self.radius, div_count=50)

        z_vec = np.array([0, 0, 1])
        sunpath_axis = self.sunpath.calc_central_axis_vec(self.p)

        self.skysphere.tilt(z_vec, sunpath_axis)

        meshes.save(self.skysphere.mesh, "./m2.obj")

        window = Window(1200, 800)
        scene = Scene()
        scene.add_mesh("Sky sphere", self.skysphere.mesh)
        scene.add_pointcloud("Analemmas", pc, size=0.08)
        window.render(scene)


if __name__ == "__main__":
    os.system("clear")
    print("--------------------- Raytracing test started -----------------------")

    test = TestSkysphere()
    test.setup_method()
    # test.test_skysphere()
    test.test_skysphere_vis()
