import os
import trimesh
import numpy as np
import pandas as pd
import math

from dtcc_solar import utils
from dtcc_solar import data_io 
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.utils import AnalysisType, Parameters
from dtcc_solar.model import Model
from dtcc_solar.results import Results
from dtcc_solar.skydome import SkyDome
from dtcc_solar.sun_analysis import SunAnalysis
from dtcc_solar.sky_analysis import SkyAnalysis
from dtcc_solar.multi_skydomes import MultiSkyDomes
import dtcc_solar.meteo_data as meteo
from dtcc_solar import weather_data as wd

from typing import List,Any
from pprint import pp

class TestRaytracing:

    lon:float
    lat:float
    city_mesh:Any
    city_model:Model
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
        self.lon = -0.12
        self.lat = 51.5        
        self.file_name = '../data/models/CitySurfaceS.stl'
        self.w_file = '../data/weather/GBR_ENG_London.City.AP.037683_TMYx.2007-2021.clm'
        self.city_mesh = trimesh.load_mesh(self.file_name)
        self.city_model = Model(self.city_mesh)     
        self.sunpath = Sunpath(self.lat, self.lon, self.city_model.sunpath_radius)
        self.sun_analysis = SunAnalysis(self.city_model)
        self.sky_analysis = SkyAnalysis(self.city_model)
        
    def test_raytracing_sun_instant(self):

        start_date = "2019-06-01 12:00:00"
        end_date = "2019-06-01 12:00:00"
        a_type = AnalysisType.sun_raycasting
        p = Parameters(a_type, self.file_name, self.lat, self.lon, 0, 0, 3, 1, 
                       False, start_date, end_date, self.w_file)

        suns = utils.create_sun_dates(p.start_date, p.end_date)    
        suns = wd.get_weather_data(p, suns)                               
        suns = self.sunpath.get_suns_positions(suns) 

        self.results = Results(suns, self.city_model.f_count)   
        self.sun_analysis.execute_raycasting_iterative(suns, self.results)
        
        face_sun_angles = self.results.res_list[0].face_sun_angles
        face_in_sun = self.results.res_list[0].face_in_sun
        is_error = False

        if np.sum(face_sun_angles) == 0.0 or np.sum(face_in_sun) == 0.0:
            is_error = True        

        assert not is_error
    
    def test_raytracing_sun_iterative(self):

        start_date = "2019-06-01 11:00:00"
        end_date = "2019-06-01 15:00:00"
        a_type = AnalysisType.sun_raycasting
        p = Parameters(a_type, self.file_name, self.lat, self.lon, 0, 0, 3, 1, 
                       False, start_date, end_date, self.w_file)

        suns = utils.create_sun_dates(p.start_date, p.end_date)
        pp(suns)    
        suns = wd.get_weather_data(p, suns)                               
        pp(suns)
        suns = self.sunpath.get_suns_positions(suns)

        print("Number of suns:" + str(len(suns))) 

        pp(suns)   

        self.results = Results(suns, self.city_model.f_count)   
        self.sun_analysis.execute_raycasting_iterative(suns, self.results)
        
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

        start_date = "2019-06-01 12:00:00"
        end_date = "2019-06-01 12:00:00"
        a_type = AnalysisType.sky_raycasting
        p = Parameters(a_type, self.file_name, self.lat, self.lon, 0, 0, 3, 1, 
                       False, start_date, end_date, self.w_file)

        suns = utils.create_sun_dates(p.start_date, p.end_date)    
        suns = wd.get_weather_data(p, suns)                               
        suns = self.sunpath.get_suns_positions(suns)
        
        self.results = Results(suns, self.city_model.f_count)   
        self.sky_analysis.execute_raycasting_iterative(suns, self.results)
        
        face_in_sky = self.results.res_acum.face_in_sky
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
                       False, start_date, end_date, self.w_file)

        suns = utils.create_sun_dates(p.start_date, p.end_date)    
        suns = wd.get_weather_data(p, suns)                               
        suns = self.sunpath.get_suns_positions(suns) 
        
        self.results = Results(suns, self.city_model.f_count)   
        self.sky_analysis.execute_raycasting_iterative(suns, self.results)
        
        res_list = self.results.res_list
        sky_irradiance = self.results.res_acum.face_irradiance_di
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

    os.system('clear')    
    print("--------------------- Raytracing test started -----------------------")

    test = TestRaytracing()
    test.setup_method()
    #test.test_raytracing_sun_instant()
    #test.test_raytracing_sun_iterative()
    #test.test_raytracing_sky_instant()
    test.test_raytracing_sky_iterative()




