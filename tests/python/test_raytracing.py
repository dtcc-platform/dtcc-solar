import os
import trimesh
import numpy as np
import pandas as pd
import math

from dtcc_solar import utils
from dtcc_solar import data_io 
from dtcc_solar.sunpath import Sunpath
from dtcc_solar.solarviewer import SolarViewer
from dtcc_solar.utils import ColorBy, AnalysisType
from dtcc_solar.model import Model
from dtcc_solar.results import Results
from dtcc_solar.data_io import Parameters
from dtcc_solar.skydome import SkyDome
from dtcc_solar.sun_analysis import SunAnalysis
from dtcc_solar.sky_analysis import SkyAnalysis
from dtcc_solar.combind_analysis import CombinedAnalysis
from dtcc_solar.multi_skydomes import MultiSkyDomes

import dtcc_solar.meteo_data as meteo

from typing import Any
from pprint import pp

class TestRaytracing:

    city_mesh:Any
    city_model:Model
    city_results:Results
    skydome:SkyDome 
    multi_skydomes:MultiSkyDomes
    sun_analysis: SunAnalysis
    sky_analysis: SkyAnalysis
    com_analysis: CombinedAnalysis
    sunpath: Sunpath
    file_name = "data/models/CitySurfaceS.stl"

    def setup_method(self):        
        self.sun_vec = np.array([ 0.49955759,  0.72086837, -0.48040713])
        self.city_mesh = trimesh.load_mesh(self.file_name)
        self.city_model = Model(self.city_mesh)
        self.city_results = Results(self.city_model)
        self.skydome = SkyDome(self.city_model.dome_radius)
        self.multi_skydomes = MultiSkyDomes(self.skydome)
        self.sun_analysis = SunAnalysis(self.city_model, self.city_results)
        self.sky_analysis = SkyAnalysis(self.city_model, self.city_results, self.skydome, self.multi_skydomes)
        self.com_analysis = CombinedAnalysis(self.city_model, self.city_results, self.skydome, self.sun_analysis, self.sky_analysis)


    def test_raytracing_instant_sun(self):
        self.sun_analysis.execute_raycasting(self.sun_vec)
        face_sun_angles = self.city_results.get_face_sun_angles()
        face_in_sun = self.city_results.get_face_in_sun()
        is_error = False

        for angle in face_sun_angles:
            if angle > 2 * math.pi:
                is_error = True
                break        

        if np.sum(face_sun_angles) == 0.0 or np.sum(face_in_sun) == 0.0:
            is_error = True        

        assert not is_error
    
    def test_raytracing_instant_sky(self):
        is_error = False
        dict_keys = ['2019-03-30T09:00:00']
        w_data = {dict_keys[0]: {'normal_irradiance': 570.0, 'horizontal_irradiance': 181.0}}
        self.sky_analysis.execute_raycasting(self.sun_vec, dict_keys, w_data)
        sky_irr_dict = self.city_results.get_sky_irradiance_dict()
        
        for key in sky_irr_dict:
            if np.sum(sky_irr_dict[key]) == 0.0:
                is_error = True

        assert not is_error

    def test_raytracing_iterative(self):
        #Get multiple solar positions for iterative analysis
        lat = 51.5
        lon = -0.12 
        self.sunpath = Sunpath(lat, lon, self.city_model.sunpath_radius)  
        time_from = pd.to_datetime('2019-03-30 07:00:00')
        time_to = pd.to_datetime('2019-03-30 21:00:00')
        [w_data, dict_keys] = meteo.get_data_from_api_call(lon, lat, time_from, time_to)
        sun_positions = self.sunpath.get_multiple_suns(dict_keys)
        [sun_positions, dict_keys] = self.sunpath.remove_sun_under_horizon(self.city_model.horizon_z, sun_positions, dict_keys)
        sun_vectors_dict = utils.get_sun_vecs_dict_from_sun_pos(sun_positions, self.city_model.origin, dict_keys)
        self.sun_analysis.execute_raycasting_iterative(sun_vectors_dict, dict_keys, w_data)
        self.sun_analysis.set_city_mesh_out()
        assert self.city_results.get_face_in_sun_dict

    def run_instant(self):
        #sky_analysis.execute_raycasting_some(sun_vec)
        #sky_analysis.set_city_mesh_out()
        #sky_analysis.set_dome_mesh_out()
        pass

if __name__ == "__main__":

    os.system('clear')    
    print("--------------------- Raytracing test started -----------------------")

    test = TestRaytracing()
    test.setup_method()
    test.test_raytracing_iterative()




