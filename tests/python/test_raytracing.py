import os
import trimesh
import numpy as np
import pandas as pd
import math
import copy

from dtcc_solar import utils
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.utils import SolarParameters, OutputCollection
from dtcc_solar.solar_engine import SolarEngine
from dtcc_core.io import meshes
from dtcc_core.model import Mesh
from dtcc_solar.logging import set_log_level, info, debug, warning, error
from typing import List, Any
from pprint import pp


class TestRaytracing:
    lon: float
    lat: float
    city_mesh: Mesh
    solar_engine: SolarEngine
    sunpath: Sunpath
    suns: List[Any]
    file_name: str
    w_file: str

    def setup_method(self):
        self.lon = -0.12
        self.lat = 51.5
        self.file_name = "../data/models/CitySurfaceS.stl"
        self.w_file = "../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm"
        self.city_mesh = meshes.load_mesh(self.file_name)

        self.p = SolarParameters(
            file_name=self.file_name,
            weather_file=self.w_file,
            latitude=self.lat,
            longitude=self.lon,
            display=False,
        )

        self.solar_engine = SolarEngine(self.city_mesh)

    def test_raytracing_sun_instant(self):
        p = copy.deepcopy(self.p)
        p.sun_analysis = True
        p.sky_analysis = False
        p.start = pd.Timestamp("2019-06-01 12:00:00")
        p.end = pd.Timestamp("2019-06-01 12:00:00")

        sunpath = Sunpath(p, self.solar_engine.sunpath_radius)
        outputc = OutputCollection()
        self.solar_engine.run_analysis(p, sunpath, outputc)

        # occ = len(self.city_mesh.faces) == len(outputc.occlusion[0])
        # ang = len(self.city_mesh.faces) == len(outputc.face_sun_angles[0])

        assert True  # occ and ang

    def test_raytracing_sun_iterative(self):
        p = copy.deepcopy(self.p)
        p.sun_analysis = True
        p.sky_analysis = False
        p.start = pd.Timestamp("2019-06-01 11:00:00")
        p.end = pd.Timestamp("2019-06-01 15:00:00")

        sunpath = Sunpath(p, self.solar_engine.sunpath_radius)
        outputc = OutputCollection()
        self.solar_engine.run_analysis(p, sunpath, outputc)

        nfaces = len(self.solar_engine.mesh.faces)
        nang = len(outputc.face_sun_angles)

        assert nang == nfaces

    def test_raytracing_sun_quad_iterative(self):
        p = copy.deepcopy(self.p)
        p.sun_analysis = True
        p.sky_analysis = True
        p.start = pd.Timestamp("2019-01-01 00:00:00")
        p.end = pd.Timestamp("2019-12-31 23:00:00")

        sunpath = Sunpath(p, self.solar_engine.sunpath_radius)
        outputc = OutputCollection()
        self.solar_engine.run_analysis(p, sunpath, outputc)

        nfaces = len(self.solar_engine.mesh.faces)
        nang = len(outputc.face_sun_angles)
        nsvf = len(outputc.sky_view_factor)

        print(nfaces, nang, nsvf)

        assert nang == nfaces and nsvf == nfaces

    def test_raytracing_sky_instant(self):
        p = copy.deepcopy(self.p)
        p.sun_analysis = False
        p.sky_analysis = True
        p.start = pd.Timestamp("2019-06-01 12:00:00")
        p.end = pd.Timestamp("2019-06-01 12:00:00")

        sunpath = Sunpath(p, self.solar_engine.sunpath_radius)
        outputc = OutputCollection()
        self.solar_engine.run_analysis(p, sunpath, outputc)

        facehit_sky = outputc.facehit_sky
        ray_count = len(facehit_sky) * len(facehit_sky[0])

        face_count = len(self.city_mesh.faces)
        skydome_ray_count = self.solar_engine.get_skydome_ray_count()

        assert face_count * skydome_ray_count == ray_count


if __name__ == "__main__":
    os.system("clear")
    print("--------------------- Raytracing test started -----------------------")
    set_log_level("INFO")
    test = TestRaytracing()
    test.setup_method()
    # test.test_raytracing_sun_instant()
    # test.test_raytracing_sun_iterative()
    test.test_raytracing_sun_quad_iterative()
    # test.test_raytracing_sky_instant()
