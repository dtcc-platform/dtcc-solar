import os
import trimesh
import numpy as np
import pandas as pd
import math

from dtcc_solar import utils
from dtcc_solar import data_io 
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.viewer import Viewer
from dtcc_solar.utils import ColorBy, AnalysisType, Mode
from dtcc_solar.model import Model
from dtcc_solar.results import Results
from dtcc_solar.data_io import Parameters
from dtcc_solar.skydome import SkyDome
from dtcc_solar.sun_analysis import SunAnalysis
from dtcc_solar.sky_analysis import SkyAnalysis
from dtcc_solar.multi_skydomes import MultiSkyDomes

from dtcc_solar.scripts.main import get_sun_and_sky, get_weather_data

import dtcc_solar.meteo_data as meteo

from typing import List,Any
from pprint import pp

class TestRaytracing:

    lon:float
    lat:float
    city_mesh:Any
    city_model:Model
    city_results:Results
    sunpath:Sunpath
    skydome:SkyDome 
    multi_skydomes:MultiSkyDomes
    sun_analysis: SunAnalysis
    sky_analysis: SkyAnalysis
    sunpath: Sunpath
    suns: List[Any]
    file_name:str
    w_file:str 

    def setup_method(self):
        self.lon = 51.2
        self.lat = -0.12        
        self.file_name = '../data/models/CitySurfaceS.stl'
        self.w_file = '../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm'
        self.sun_vec = np.array([ 0.49955759,  0.72086837, -0.48040713])
        self.city_mesh = trimesh.load_mesh(self.file_name)
        self.city_model = Model(self.city_mesh)
        self.city_results = Results(self.city_model)
        self.skydome = SkyDome(self.city_model.dome_radius)
        self.sunpath = Sunpath(self.lat, self.lon, self.city_model.sunpath_radius)
        self.multi_skydomes = MultiSkyDomes(self.skydome)
        self.sun_analysis = SunAnalysis(self.city_model, self.city_results)
        self.sky_analysis = SkyAnalysis(self.city_model, self.city_results, self.skydome, self.multi_skydomes)
        
    def test_raytracing_sun_instant(self):

        one_date = "2019-06-01 12:00:00"
        a_type = AnalysisType.sun_raycasting
        print(self.file_name)
        p = Parameters(a_type, self.file_name, self.lat, self.lon, 0, 0, 3, 1, 
                       False, one_date, one_date, one_date, self.w_file, 1)

        [suns, skys] = get_sun_and_sky(p, self.sunpath)
        self.sun_analysis.execute_raycasting_iterative(suns)
        
        face_sun_angles = self.city_results.get_face_sun_angles()
        face_in_sun = self.city_results.get_face_in_sun()
        is_error = False

        if np.sum(face_sun_angles) == 0.0 or np.sum(face_in_sun) == 0.0:
            is_error = True        

        assert not is_error
    
    def test_raytracing_sun_iterative(self):

        start_date = "2019-06-01 11:00:00"
        end_date = "2019-06-01 15:00:00"
        a_type = AnalysisType.sun_raycasting
        p = Parameters(a_type, self.file_name, self.lat, self.lon, 0, 0, 3, 1, 
                       False, start_date, start_date, end_date, self.w_file, 2)

        [suns, skys] = get_sun_and_sky(p, self.sunpath)
        self.sun_analysis.execute_raycasting_iterative(suns)
        
        face_sun_angles_dict = self.city_results.get_face_sun_angles_dict()
        face_in_sun_dict = self.city_results.get_face_in_sun_dict()
        is_error = False

        for key in face_sun_angles_dict:
            fsa = face_sun_angles_dict[key]
            if np.sum(fsa) == 0.0:
                print("Test failed!")
                is_error = True
                break        
        
        for key in face_in_sun_dict:
            fis = face_in_sun_dict[key]
            if np.sum(fis) == 0.0:
                print("Test failed!")
                is_error = True
                break        

        assert not is_error


    def test_raytracing_sky_instant(self):

        one_date = "2019-06-01 12:00:00"
        a_type = AnalysisType.sky_raycasting
        p = Parameters(a_type, self.file_name, self.lat, self.lon, 0, 0, 3, 1, 
                       False, one_date, one_date, one_date, self.w_file, 1)

        [suns, skys] = get_sun_and_sky(p, self.sunpath)
        self.sky_analysis.execute_raycasting_iterative(skys)
        
        face_in_sky = self.city_results.get_face_in_sky()
        is_error = False

        if np.sum(face_in_sky) == 0.0:
            print("Test failed!")
            is_error = True        

        assert not is_error

    def test_raytracing_sky_iterative(self):
        
        start_date = "2019-06-01 11:00:00"
        end_date = "2019-06-01 15:00:00"
        a_type = AnalysisType.sun_raycasting
        p = Parameters(a_type, self.file_name, self.lat, self.lon, 0, 0, 3, 1, 
                       False, start_date, start_date, end_date, self.w_file, 2)

        [suns, skys] = get_sun_and_sky(p, self.sunpath)
        self.sky_analysis.execute_raycasting_iterative(skys)
        
        sky_irradiance = self.city_results.get_sky_irradiance_dict()
        is_error = False

        pp(skys)
        
        si_sum = 0
        for key in sky_irradiance:
            si = sky_irradiance[key]
            si_sum += np.sum(si)

        if si_sum == 0.0:
            print("Test failed. No sky irradiance recorded. Check dates and time!")
            is_error = True    
            
        assert not is_error

if __name__ == "__main__":

    os.system('clear')    
    print("--------------------- Raytracing test started -----------------------")

    test = TestRaytracing()
    test.setup_method()
    #test.test_raytracing_sun_instant()
    #test.test_raytracing_sun_iterative()
    #test.test_raytracing_sky_instant()
    test.test_raytracing_sky_iterative()




