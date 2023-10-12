import os
import trimesh
import numpy as np
import pandas as pd
import math

from dtcc_solar import utils
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.utils import AnalysisType, SolarParameters, DataSource
from dtcc_solar.solar_engine import SolarEngine
from dtcc_solar.results import Results
from dtcc_solar.skydome import SkyDome
import dtcc_solar.data_meteo as meteo
from dtcc_solar import weather_data as weather

from typing import List, Any
from pprint import pp


class TestRaytracing:
    lon: float
    lat: float
    city_mesh: Any
    solar_engine: SolarEngine
    sunpath: Sunpath
    skydome: SkyDome
    sunpath: Sunpath
    suns: List[Any]
    file_name: str
    w_file: str

    def setup_method(self):
        self.lon = -0.12
        self.lat = 51.5
        self.file_name = "../data/models/CitySurfaceS.stl"
        self.w_file = "../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm"
        self.city_mesh = trimesh.load_mesh(self.file_name)
        self.solar_engine = SolarEngine(self.city_mesh)
        self.sunpath = Sunpath(self.lat, self.lon, self.solar_engine.sunpath_radius)
        self.skydome = SkyDome(self.solar_engine.dome_radius)

        self.p = SolarParameters(
            file_name=self.file_name,
            weather_file=self.w_file,
            latitude=self.lat,
            longitude=self.lon,
        )

    def test_raytracing_sun_instant(self):
        self.p.a_type = AnalysisType.sun_raycasting
        self.p.start_date = "2019-06-01 12:00:00"
        self.p.end_date = "2019-06-01 12:00:00"

        self.suns = self.sunpath.create_suns(self.p)

        self.results = Results(self.suns, self.solar_engine.f_count)
        self.solar_engine.sun_raycasting(self.suns, self.results)

        face_sun_angles = self.results.res_list[0].face_sun_angles
        face_in_sun = self.results.res_list[0].face_in_sun
        is_error = False

        if np.sum(face_sun_angles) == 0.0 or np.sum(face_in_sun) == 0.0:
            is_error = True

        assert not is_error

    def test_raytracing_sun_iterative(self):
        self.p.a_type = AnalysisType.sun_raycasting
        self.p.start_date = "2019-06-01 11:00:00"
        self.p.end_date = "2019-06-01 15:00:00"

        self.suns = self.sunpath.create_suns(self.p)
        self.results = Results(self.suns, self.solar_engine.f_count)
        self.solar_engine.sun_raycasting(self.suns, self.results)

        res_list = self.results.res_list
        is_error = False

        for res in res_list:
            fsa = res.face_sun_angles
            fis = res.face_in_sun
            print(np.sum(fsa))
            if np.sum(fsa) == 0.0 or np.sum(fis) == 0.0:
                print("Test failed!")
                is_error = True
                break

        assert not is_error

    def test_raytracing_sky_instant(self):
        self.p.a_type = AnalysisType.sky_raycasting
        self.p.start_date = "2019-06-01 12:00:00"
        self.p.end_date = "2019-06-01 12:00:00"

        self.suns = self.sunpath.create_suns(self.p)
        self.suns = weather.append_weather_data(self.p, self.suns)

        self.results = Results(self.suns, self.solar_engine.f_count)
        self.solar_engine.sky_raycasting(self.suns, self.results, self.skydome)

        face_in_sky = self.results.res_acum.face_in_sky
        is_error = False
        if np.sum(face_in_sky) == 0.0:
            print("Test failed!")
            is_error = True

        assert not is_error

    def test_raytracing_sky_iterative(self):
        self.p.a_type = AnalysisType.sun_raycasting
        self.p.start_date = "2019-06-01 11:00:00"
        self.p.end_date = "2019-06-01 15:00:00"

        self.suns = self.sunpath.create_suns(self.p)
        self.suns = weather.append_weather_data(self.p, self.suns)

        self.results = Results(self.suns, self.solar_engine.f_count)
        self.solar_engine.sky_raycasting(self.suns, self.results, self.skydome)

        res_list = self.results.res_list
        is_error = False

        di_sum = 0
        for res in res_list:
            di = res.face_irradiance_di
            di_sum += np.sum(di)

        if di_sum == 0.0:
            print("Test failed. No sky irradiance recorded. Check dates and time!")
            is_error = True

        assert not is_error


if __name__ == "__main__":
    os.system("clear")
    print("--------------------- Raytracing test started -----------------------")

    test = TestRaytracing()
    test.setup_method()
    # test.test_raytracing_sun_instant()
    # test.test_raytracing_sun_iterative()
    # test.test_raytracing_sky_instant()
    test.test_raytracing_sky_iterative()
