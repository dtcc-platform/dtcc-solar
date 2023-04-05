import numpy as np
import dtcc_solar.raycasting as raycasting
import dtcc_solar.mesh_compute as mc
from dtcc_solar import utils

from dtcc_solar.sun_analysis import SunAnalysis
from dtcc_solar.sky_analysis import SkyAnalysis
from dtcc_solar.results import Results
from dtcc_solar.skydome import SkyDome
from dtcc_solar.model import Model

class CombinedAnalysis:

    sun_analysis : SunAnalysis
    sky_analysis : SkyAnalysis
    results : Results
    skydome : SkyDome
    model : Model

    def __init__(self, model, results, skydome, sun_analysis, sky_analysis):
        self.model = model
        self.results = results
        self.skydome = skydome
        self.sun_analysis = sun_analysis
        self.sky_analysis = sky_analysis 
        self.flux = 1000 #Watts per m2

    def execute(self, sun_vectors_dict, dict_keys, w_data):
        self.sun_analysis.execute_raycasting_iterative(sun_vectors_dict, dict_keys, w_data)    
        self.sky_analysis.execute_raycasting(sun_vectors_dict, dict_keys, w_data)
        
    def set_city_mesh_out(self):
        self.results.set_city_mesh_out(self.model.city_mesh)
